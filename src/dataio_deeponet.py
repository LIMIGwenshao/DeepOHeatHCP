import torch, copy, os
import numpy as np
from torch.utils.data import Dataset

from src.dataio_utils import fixed_mesh_grid_3d
from src.geometry_deeponet import (
    create_stacking_cuboidal_geometry,
    fetch_mesh_data,
)
import matplotlib.pyplot as plt

# The dataio classes used in training DeepOHeat: 2D power map

class DeepONetMeshDataIO(Dataset):

    def __init__(self, domains_list, global_params, dim=2, var=1, len_scale=0.3):
        super().__init__()
        self.geometry = create_stacking_cuboidal_geometry( #最后返回的是root的cubic
            domains_list, dim=dim, mesh=True
        )# dataset 核心
        self.pde_params = global_params["pde_params"]
        self.loss_fun_type = global_params["loss_fun_type"]
        self.num_params_per_epoch = global_params["num_params_per_epoch"]
        self.dim = dim
        self.mode = "train"
        self.var = var
        self.len_scale = len_scale

    def __len__(self):
        return self.num_params_per_epoch

    def __getitem__(self, idx): #获取数据的函数，使用next(iter(self))调用
        return fetch_mesh_data(
            self.geometry,
            self.mode,
            dim=self.dim,
            var=self.var,
            len_scale=self.len_scale,
        )

    # def draw_power_map(self, model_dir):
    #     """
    #     绘制3D温度分布图
    #     """
    #     import matplotlib.pyplot as plt
    #     from mpl_toolkits.mplot3d import Axes3D
    #
    #     # 获取训练数据
    #     train_data, _, _ = self.train()
    #     train_coords, train_sensors = train_data.values()
    #
    #     # 创建3D图形
    #     fig = plt.figure(figsize=(12, 8))
    #     ax = fig.add_subplot(111, projection='3d')
    #
    #     # 使用train_coords作为3D坐标
    #     x = train_coords[:, 0]
    #     y = train_coords[:, 1]
    #     z = train_coords[:, 2]
    #
    #     # 使用train_sensors作为温度值
    #     temperature = train_sensors[0, :].reshape(-1)
    #
    #     # 绘制3D散点图，使用内置的'jet'颜色映射
    #     scatter = ax.scatter(
    #         x, y, z,
    #         c=temperature,
    #         cmap='jet',  # 使用内置的jet颜色映射
    #         s=10,
    #         alpha=0.8
    #     )
    #
    #     # 添加颜色条
    #     cbar = plt.colorbar(scatter)
    #     cbar.set_label('Temperature')
    #
    #     # 设置坐标轴标签
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #
    #     # 设置标题
    #     plt.title('3D Temperature Distribution')
    #
    #     # 调整视角
    #     ax.view_init(elev=30, azim=45)
    #
    #     # 保存图像
    #     plt.savefig(os.path.join(model_dir, '3d_temperature.png'), dpi=300, bbox_inches='tight')
    #     plt.close()


    # def visualize_3d_temperature(self, model_dir):
    #     """
    #     绘制3D温度分布图
    #
    #     参数:
    #     model_dir: 模型保存目录
    #     train_data: 训练数据
    #     geometry: 几何对象
    #     """
    #     train_data, _, _ = self.train()
    #     train_coords, train_sensors = train_data.values()
    #     train_sensor_power = train_sensors[0, :].reshape(-1)
    #     # 创建输出目录
    #     fig_dir = os.path.join(model_dir, "figure")
    #     os.makedirs(fig_dir, exist_ok=True)
    #
    #     # 获取训练数据
    #     train_coords, train_sensors = train_data.values()
    #     train_sensor_power = train_sensors[0, :].reshape(-1)
    #
    #     # 获取顶部边界点
    #     self.geometry.update_set()
    #     top_idx_set = self.geometry.boundaries_set["top"]
    #     top_coords = train_coords[top_idx_set, :]
    #     top_power = train_sensor_power[top_idx_set]
    #
    #     # 创建3D图形
    #     fig = plt.figure(figsize=(12, 8))
    #     ax = fig.add_subplot(111, projection='3d')
    #
    #     # 绘制3D散点图
    #     scatter = ax.scatter3D(
    #         top_coords[:, 0],
    #         top_coords[:, 1],
    #         top_coords[:, 2],
    #         c=top_power,
    #         cmap='jet',
    #         s=10,
    #         alpha=0.8
    #     )
    #
    #     # 添加颜色条
    #     cbar = fig.colorbar(
    #         scatter,
    #         ax=ax,
    #         shrink=0.5,
    #         aspect=5,
    #         ticks=np.linspace(top_power.min(), top_power.max(), 10)
    #     )
    #     cbar.set_label('Temperature')
    #
    #     # 设置坐标轴标签
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #
    #     # 设置标题
    #     plt.title('3D Temperature Distribution')
    #
    #     # 调整视角
    #     ax.view_init(elev=30, azim=45)
    #
    #     # 保存图像
    #     plt.savefig(os.path.join(fig_dir, 'temperature_3d.png'), dpi=300, bbox_inches='tight')
    #     plt.close("all")
    #

    def draw_power_map(self, model_dir):
        train_data, _, _ = self.train()
        train_coords, train_sensors = train_data.values()
        train_sensor_power = train_sensors[0, :].reshape(-1)
        # print(train_sensor_power.shape)
        # print(train_sensor_power)
        starts, ends, num_intervals, _, _ = self.geometry.domain["geometry"].values()
        mesh = fixed_mesh_grid_3d(
            starts=starts[:2], ends=ends[:2], num_intervals=num_intervals[:2]
        )

        fig_dir = os.path.join(model_dir, "figure")

        fig = plt.figure()
        # plt.scatter(mesh[:, 0], mesh[:, 1])
        plt.scatter(mesh[:, 0], mesh[:, 1], c=train_sensor_power, cmap="jet")
        plt.colorbar(
            ticks=np.linspace(train_sensor_power.min(), train_sensor_power.max(), 10)
        )
        plt.savefig(os.path.join(fig_dir, "train_sensor.png"))
        plt.close("all")

        self.geometry.update_set()
        top_idx_set = self.geometry.boundaries_set["top"]
        top_coords = train_coords[top_idx_set, :]

        fig = plt.figure()
        ax = plt.axes(projection="3d")
        sctt = ax.scatter3D(
            top_coords[:, 0],
            top_coords[:, 1],
            top_coords[:, 2],
            c=train_sensor_power,
            cmap="jet",
        )

        fig.colorbar(
            sctt,
            ax=ax,
            shrink=0.5,
            aspect=5,
            ticks=np.linspace(train_sensor_power.min(), train_sensor_power.max(), 10),
        )
        plt.savefig(os.path.join(fig_dir, "coords_power_map.png"))

        eval_data = self.eval()
        _, power_map = eval_data.values()
        power_map = power_map[0, :].reshape(-1)

        fig = plt.figure()
        #plt.scatter(mesh[:, 0], mesh[:, 1])
        plt.scatter(mesh[:, 0], mesh[:, 1], c=power_map, cmap="jet")
        plt.colorbar(ticks=np.linspace(power_map.min(), power_map.max(), 10))
        plt.savefig(os.path.join(fig_dir, "eval_power_map.png"))
        plt.close("all")

    def train(self):
        self.mode = "train"
        sensors, coords, conductivity = next(iter(self))     #return (#    np.concatenate(sensors_list, 0), np.concatenate(tensor_list, 0), np.concatenate(conductivity_list, 0),)
        sensors = sensors.reshape(1, -1).repeat(coords.shape[0], 0) #这里改变了结构用于模型输入
        #print(sensors.shape)
        return (
            {"coords": torch.tensor(coords), "beta": torch.tensor(sensors)},
            torch.tensor(conductivity),
            copy.deepcopy(self.geometry),#深拷贝
        )

    def eval(self):
        self.mode = "eval"
        sensors, coords, _ = next(iter(self))
        sensors = sensors.reshape(1, -1).repeat(coords.shape[0], 0)

        return {"coords": torch.tensor(coords), "beta": torch.tensor(sensors)}

    #return (
    #    np.concatenate(sensors_list, 0),
    #    np.concatenate(tensor_list, 0),
    #    np.concatenate(conductivity_list, 0),
    #)
