import torch
import copy
from torch.utils.data import Dataset
from src.geometry_mesh import create_stacking_cuboidal_geometry,fetch_mesh_data


class CuboidGeometryDataIO(Dataset):

    def __init__(self, domains_list, global_params, dim=2, var=1, len_scale=0.3, beta_as_input=False):
        super().__init__()
        self.geometry = create_stacking_cuboidal_geometry(domains_list, dim=dim, mesh=True)
        self.pde_params = global_params["pde_params"]
        self.loss_fun_type = global_params["loss_fun_type"]
        self.num_params_per_epoch = global_params["num_params_per_epoch"]

        self.beta_as_input = beta_as_input # true/false
        self.dim = dim
        self.mode = "train"
        self.var = var
        self.len_scale = len_scale
        self.args = []#特有的

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

