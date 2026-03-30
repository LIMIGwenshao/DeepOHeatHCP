import torch
import numpy as np
from numpy.ma.extras import average
from ordered_set import OrderedSet
from src.diff_operator import *
import matplotlib.pyplot as plt
from src.geometry import *


def cal_vec_loss(loss_fun_type, vec, weight=1):
    if loss_fun_type == "mse":
        loss = torch.nn.functional.mse_loss(vec, torch.zeros_like(vec))
    elif loss_fun_type == "norm":
        loss = vec.norm()
    elif loss_fun_type == "squared_norm":
        loss = vec.norm() ** 2
    elif loss_fun_type == "msn":
        loss = vec.norm() ** 2 / len(vec)

    return loss * weight if not torch.isnan(loss).item() else 0

#这些函数传入的参数都是默认值，真实影响训练的是刚开始定义的参数，这些默认值会被覆盖。
def loss_adiabatics(loss_fun_type, u, jac, u_laplace, idx, dim, weight=1):
    # vec = jac[..., 0, dim][idx].squeeze()
    vec = jac[..., dim][idx].squeeze()
    return cal_vec_loss(loss_fun_type, vec, weight)


def loss_dirichelet(loss_fun_type, u, jac, u_laplace, idx, value, weight=1):
    vec = (u[idx, :] - torch.ones_like(u[idx, :]) * value).squeeze()
    return cal_vec_loss(loss_fun_type, vec, weight)


def loss_robin(loss_fun_type, u, jac, u_laplace, idx, dim, k, direction, weight=1):
    # vec = (u[idx, :]-0.2+direction*k*jac[..., 0, dim][idx]).squeeze()
    vec = (u[idx, :].squeeze() - 0.2 + direction * k * jac[..., dim][idx]).squeeze()
    return cal_vec_loss(loss_fun_type, vec, weight)


def loss_pde(loss_fun_type, u, jac, u_laplace, idx, k=0.2, weight=1):
    vec = (u_laplace[idx] * k).squeeze()
    return cal_vec_loss(loss_fun_type, vec, weight)


def loss_volumetric_power(loss_fun_type, u, jac, u_laplace, idx, k=0.2, value=1, weight=1):
    vec = (u_laplace[idx] * k + value * torch.ones_like(u_laplace[idx])).squeeze()
    return cal_vec_loss(loss_fun_type, vec, weight)


def loss_neumann(loss_fun_type, u, jac, u_laplace, idx, dim=2, weight=1):
    # vec = jac[..., 0, dim][idx].squeeze()
    vec = jac[..., dim][idx].squeeze()
    return cal_vec_loss(loss_fun_type, vec, weight)


def loss_surface_power(loss_fun_type, u, jac, u_laplace, idx, dim=2, value=1, weight=1):
    # vec = (jac[..., 0, dim][idx] - torch.ones_like(jac[..., 0, dim][idx])*value).squeeze()
    vec = (jac[..., dim][idx] - torch.ones_like(jac[..., dim][idx]) * value).squeeze()
    return cal_vec_loss(loss_fun_type, vec, weight)


def loss_interface_temperature(loss_fun_type, u, jac, u_laplace, idx_parent, idx, dim=2, weight=1):
    # print(dim)
    # print(weight)
    # print(idx)
    # print(idx_parent)
    temperature_parent = u[idx_parent, :].squeeze()
    # print(temperature_parent)
    temperature = u[idx, :].squeeze()
    # print(temperature)
    vec = (temperature_parent.mean() - temperature.mean()).abs()
    vec = vec.unsqueeze(0)  # 变成 shape=[1]，与 cal_vec_loss 输入格式统一
    return cal_vec_loss(loss_fun_type, vec, weight)


def loss_interface_convection(loss_fun_type, u, jac, u_laplace, idx_parent, idx, k_parent, k, dim=2, weight=1):
    grad_parent = jac[..., dim][idx_parent].squeeze()
    grad = jac[..., dim][idx].squeeze()
    flux_parent = 0.2 * k_parent * grad_parent   # 父域通量（正向）
    flux_child = 0.2 * k * grad      # 子域通量（视为方向一致）
    # print(flux_parent)
    # print(flux_child)
    avg_flux_parent = flux_parent.mean()
    avg_flux_child = flux_child.mean()
    vec = (avg_flux_parent - avg_flux_child).abs().unsqueeze(0)  # 标量转成1D张量，方便损失计算
    return cal_vec_loss(loss_fun_type, vec, weight)


def find_boundaries_endpoints(starts, ends):
    boundaries_dict = dict(
        front=dict(starts=starts, ends=[ends[0], starts[1], ends[2]]),
        back=dict(starts=[starts[0], ends[1], starts[2]], ends=ends),
        left=dict(starts=starts, ends=[starts[0], ends[1], ends[2]]),
        right=dict(starts=[ends[0], starts[1], starts[2]], ends=ends),
        bottom=dict(starts=starts, ends=[ends[0], ends[1], starts[2]]),
        top=dict(starts=[starts[0], starts[1], ends[2]], ends=ends),
    )

    return boundaries_dict


def loss_fun_geometry_init(dataset):
    pde_params = dataset.pde_params
    loss_fun_type = dataset.loss_fun_type

    bc_loss_fun_dict = {
        "pde": loss_pde,
        "htc": loss_robin,
        "adiabatics": loss_adiabatics,
        "volumetric_power": loss_volumetric_power,
        "surface_power": loss_surface_power,
        "neumann": loss_neumann,
        "dirichelet": loss_dirichelet,
        "interface_temperature": loss_interface_temperature,
        "interface_convection": loss_interface_convection,
    }

    def loss_fn(u, jac, u_laplace, geometry):

        loss_dict = {
            "pde": 0,
            "htc": 0,
            "adiabatics": 0,
            "volumetric_power": 0,
            "surface_power": 0,
            "neumann": 0,
            "dirichelet": 0,
            "interface_temperature":0,
            "interface_convection": 0,
        }
        pde_set_list = []

        def bc_loss_cal(boundary_dict, boundary_idx):
            loss_type = boundary_dict["type"]#返回边界名称，在初始化时编辑
            loss_fun = bc_loss_fun_dict[loss_type]#返回函数名称，即由名称对应到函数名
            loss_dict[loss_type] += loss_fun(
                loss_fun_type,
                u,
                jac,
                u_laplace,
                boundary_idx,#对应set集，即这里存放的是索引
                *boundary_dict["params"].values(),
            )

        def single_node_loss_fun(node):
            node.update_set()#主要进行的是一个连接处点的剔除
            pde_set_list.append(node.pde_set)

            if node.domain_step["power"]["bc"]:#首先计算功率损失
                for power_id, power_i in node.domain_step["power"]["power_map"].items():
                    bc_loss_cal(power_i, node.power_points_set_dict[power_id])

            for boundary_name, boundary_set in node.boundaries_set.items():
                boundary_dict = node.domain_step[boundary_name]#模型初始化定义的字典

                if not boundary_dict["bc"]:
                    continue

                if boundary_dict["type"] == "interface": #连续性边界条件特殊计算
                    parent_node = node.to_parent()
                    parent_node_interface_set = parent_node.interface_set #得到父节点的接触面点的索引
                    interface_set = node.interface_set #得到该节点的接触面点的索引
                    parent_conductivity = parent_node.conductivity
                    conductivity = node.conductivity
                    #温度连续——平均
                    loss_dict["interface_temperature"] += loss_interface_temperature(
                        loss_fun_type,
                        u,
                        jac,
                        u_laplace,
                        parent_node_interface_set,
                        interface_set,
                        *boundary_dict["params"].values(),
                    )
                    #通量连续——平均
                    loss_dict["interface_convection"] += loss_interface_convection(
                        loss_fun_type,
                        u,
                        jac,
                        u_laplace,
                        parent_node_interface_set,
                        interface_set,
                        parent_conductivity.mean(),#对所有值都相同的热导率数组求平均，结果是将numpy转换为常数
                        conductivity.mean(),
                        *boundary_dict["params"].values(),
                    )
                    continue

                bc_loss_cal(boundary_dict, boundary_set)

        iterate_over_entire_geometry(geometry, single_node_loss_fun)
        pde_set = OrderedSet().union(*pde_set_list)
        bc_loss_cal(pde_params, pde_set)#pde_set是减去power的

        return loss_dict

    return loss_fn
