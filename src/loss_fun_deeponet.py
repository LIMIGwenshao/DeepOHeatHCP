import torch
import numpy as np
from ordered_set import OrderedSet
from src.diff_operator import *
import matplotlib.pyplot as plt
from src.geometry_utils import iterate_over_entire_geometry

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


def loss_volumetric_power(
    loss_fun_type, u, jac, u_laplace, idx, k=0.2, value=1, weight=1):
    vec = (u_laplace[idx] * k + value * torch.ones_like(u_laplace[idx])).squeeze()
    return cal_vec_loss(loss_fun_type, vec, weight)


def loss_neumann(loss_fun_type, u, jac, u_laplace, idx, dim=2, weight=1):
    # vec = jac[..., 0, dim][idx].squeeze()
    vec = jac[..., dim][idx].squeeze()
    return cal_vec_loss(loss_fun_type, vec, weight)


def loss_surface_power(loss_fun_type, u, jac, u_laplace, idx, dim, value, weight=1):
    # vec = (jac[..., 0, dim][idx] - torch.ones_like(jac[..., 0, dim][idx])*value).squeeze()
    vec = (jac[..., dim][idx] - torch.ones_like(jac[..., dim][idx]) * value).squeeze()
    return cal_vec_loss(loss_fun_type, vec, weight)


def loss_arbitrary_surface_power(loss_fun_type, jac, q, idx, weight=1):
    # vec = (jac[..., 0, dim][idx] - torch.ones_like(jac[..., 0, dim][idx])*value).squeeze()
    vec = jac[..., 2][idx].squeeze() - q[idx]
    return cal_vec_loss(loss_fun_type, vec, weight)


def loss_mesh_arbitrary_surface_power(loss_fun_type, jac, q, idx, weight=1):
    # vec = (jac[..., 0, dim][idx] - torch.ones_like(jac[..., 0, dim][idx])*value).squeeze()
    # print("jac[..., 2][idx].squeeze().shape:", jac[..., 2][idx].squeeze().shape)
    # print("q.shape:", q.shape)
    vec = jac[..., 2][idx].squeeze() - q
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


def mesh_loss_fun_geometry_init(dataset):# 传递的这个函数，输入参数类型位dataset
    pde_params = dataset.pde_params
    loss_fun_type = dataset.loss_fun_type #这里定义的是平方或者均方误差

    bc_loss_fun_dict = {
        "pde": loss_pde,
        "htc": loss_robin,
        "adiabatics": loss_adiabatics,
        "volumetric_power": loss_volumetric_power,
        "surface_power": loss_surface_power,
        "neumann": loss_neumann,
        "dirichelet": loss_dirichelet,
    } # 这个字典里存放的只是函数的名称

    def top_2d_power_loss_fn(u, jac, u_laplace, beta, geometry): #因此真正计算损失值时调用的是这个函数

        loss_dict = {
            "pde": 0,
            "htc": 0,
            "adiabatics": 0,
            "volumetric_power": 0,
            "surface_power": 0,
            "neumann": 0,
            "dirichelet": 0,
        }#首先定义一个包含各种损失值的字典，并将所有值初始化为0
        pde_set_list = []#定义控制方程损失list为空

        def bc_loss_cal(boundary_dict, boundary_idx): #定义边界损失值计算函数
            loss_type = boundary_dict["type"]#首先赋值边界类型，即首先确定的是边界的类型
            loss_fun = bc_loss_fun_dict[loss_type]#然后定义边界损失函数
            loss_dict[loss_type] += loss_fun(#奖损失值计算并加起来
                loss_fun_type,
                u,
                jac,
                u_laplace,
                boundary_idx,
                *boundary_dict["params"].values(),
            )

        def single_node_loss_fun(node):#这个是计算每个节点的损失
            node.update_set() # 更新各边界点
            pde_set_list.append(node.pde_set)

            # if node.domain_step["power"]["bc"]:
            #     for power_id, power_i in node.domain_step["power"]["power_map"].items():
            #         bc_loss_cal(power_i, node.power_points_set_dict[power_id])

            for boundary_name, boundary_set in node.boundaries_set.items():#其键是边界的名字，值是该边界对应的点集
                boundary_dict = node.domain_step[boundary_name]

                #print(node.name)
                #print(node.domain_step[boundary_name])
                #if boundary_name == "bottom":# 这里修改过 原先为top
                #if boundary_name == "top" and node.is_leaf():
                #如果是3维建模吧把这里注释掉
                if boundary_name == "top":
                    power_map = beta[0, :].reshape(-1)
                    #print(f"power map shape: {power_map.shape}")
                    #print(f"power map: {power_map}")
                    loss_dict["surface_power"] += loss_mesh_arbitrary_surface_power(
                        loss_fun_type, jac, power_map, boundary_set
                    )#jac是power map的梯度，
                    continue

                if not boundary_dict["bc"]:
                    continue

                bc_loss_cal(boundary_dict, boundary_set)# 传入边界字典

        #print("signal")
        iterate_over_entire_geometry(geometry, single_node_loss_fun)# 这个函数给一个几何再给一个函数，然后再每个几何里调用这个函数
        pde_set = OrderedSet().union(*pde_set_list)
        bc_loss_cal(pde_params, list(pde_set))

        return loss_dict#

    return top_2d_power_loss_fn #返回的是这个函数
