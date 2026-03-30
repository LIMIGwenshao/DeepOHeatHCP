import torch, os, sys, matplotlib
import matplotlib.pyplot as plt
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
sys.path.append("../../")
from src import modules, dataio_deeponet
from src.utils import MyCmap

domain_0 = dict(
    domain_name=0,
    geometry=dict(
        starts=[0.0, 0.0, 0.0],
        ends=[2.0, 2.0, 0.55],
        num_intervals=[20, 20, 11],
        num_pde_points=8000, #不知道怎么确定的
        num_single_bc_points=1000, #不知道怎么确定的
    ),
    conductivity_dist=dict(uneven_conductivity=False, background_conductivity=5),
    power=dict(
        bc=True,
        num_power_points_per_volume=4,
        num_power_points_per_surface=500,
        num_power_points_per_cell=5,
        power_map=dict(
            power_0=dict(
                type="volumetric_power",
                location=dict(starts=(0, 0, 5), ends=(20, 20, 6)),
                params=dict(k=0.2, value=1, weight=1),
            )
        ),
    ),
    front=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    back=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    left=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    right=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    bottom=dict(bc=True, type="htc", params=dict(dim=2, k=0.2, direction=-1, weight=1)),
    top=dict(bc=True, type="adiabatics", params=dict(dim=2, weight=1)),
    node=dict(root=True, leaf=False, children=dict(top=[1,2,3,4])),
    parameterized=dict(variable=False),
)

domain_1 = dict(
    domain_name=1,
    geometry=dict(
        starts=[0.4, 0.4, 0.55],
        ends=[0.9, 0.9, 0.75],
        num_intervals=[5, 5, 4],
        num_pde_points=400,
        num_single_bc_points=100,
    ),
    conductivity_dist=dict(uneven_conductivity=False, background_conductivity=5),
    power=dict(bc=False),
    front=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    back=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    left=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    right=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    bottom=dict(bc=False),
    top=dict(bc=True, type="htc", params=dict(dim=2, k=0.2, direction=1, weight=1)),
    node=dict(root=False, leaf=True),
    parameterized=dict(variable=False),
)

domain_2 = dict(
    domain_name=2,
    geometry=dict(
        starts=[0.4, 1.1, 0.55],
        ends=[0.9, 1.6, 0.75],
        num_intervals=[5, 5, 4],
        num_pde_points=400,
        num_single_bc_points=100,
    ),
    conductivity_dist=dict(uneven_conductivity=False, background_conductivity=10),
    power=dict(bc=False),
    front=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    back=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    left=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    right=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    bottom=dict(bc=False),
    top=dict(bc=True, type="htc", params=dict(dim=2, k=0.2, direction=1, weight=1)),
    node=dict(root=False, leaf=True),
    parameterized=dict(variable=False),
)

domain_3 = dict(
    domain_name=3,
    geometry=dict(
        starts=[1.1, 0.4, 0.55],
        ends=[1.6, 0.9, 0.75],
        num_intervals=[5, 5, 4],
        num_pde_points=400,
        num_single_bc_points=100,
    ),
    conductivity_dist=dict(uneven_conductivity=False, background_conductivity=15),
    power=dict(bc=False),
    front=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    back=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    left=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    right=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    bottom=dict(bc=False),
    top=dict(bc=True, type="htc", params=dict(dim=2, k=0.2, direction=1, weight=1)),
    node=dict(root=False, leaf=True),
    parameterized=dict(variable=False),
)

domain_4 = dict(
    domain_name=4,
    geometry=dict(
        starts=[1.1, 1.1, 0.55],
        ends=[1.6, 1.6, 0.75],
        num_intervals=[5, 5, 4],
        num_pde_points=400,
        num_single_bc_points=100,
    ),
    conductivity_dist=dict(uneven_conductivity=False, background_conductivity=20),
    power=dict(bc=False),
    front=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    back=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    left=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    right=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    bottom=dict(bc=False),
    top=dict(bc=True, type="htc", params=dict(dim=2, k=0.2, direction=1, weight=1)),
    node=dict(root=False, leaf=True),
    parameterized=dict(variable=False),
)

domains_list = [domain_0, domain_1, domain_2, domain_3, domain_4]
global_params = {
    "loss_fun_type": "norm",
    "num_params_per_epoch": 2,
    "pde_params": dict(type="pde", params=dict(k=0.2, weight=1)),
}

print(
    "Evaluating trained DeepOHeat: parameterized top HTC BC with volumetric power defined in the middle layer"
)

device = "cuda:0"
model = modules.FFONet(
    trunk_in_features=3,
    trunk_hidden_features=128,
    branch_in_features=6012,
    branch_hidden_features=20,
    inner_prod_features=50,
    num_branch_hidden_layers=3,
    num_trunk_hidden_layers=3,
    nonlinearity="silu",
    trunk_freq=torch.pi,
    branch_freq=torch.pi,
    std=1,
    freq_trainable=True,
    device=device,
)
print("The model used for this case:", model)
var = 1 #这两个参数大概实际控制的是power map的高斯采样参数
len_scale = 0.3

root_path = "./log"
experiment_name = "test"
epoch = 50
model_dir = os.path.join(
    root_path, experiment_name, "checkpoints", "model_epoch_{}.pth".format(epoch)
)
checkpoint = torch.load(model_dir)
model.load_state_dict(checkpoint["model"])
model.eval()

figure_dir = os.path.join(
    root_path, experiment_name, "eval", "model_epoch_{}".format(epoch)
)
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

dataset = dataio_deeponet.DeepONetMeshDataIO( #数据集包括几何参数、各种参数、以及训练模式和测试模式、画图
    domains_list, global_params, dim=3, var=var, len_scale=len_scale #dim=3
)
cmap = MyCmap.get_cmap()

for sample_mode in ["low", "middle", "high"]:
    eval_data = dataset.eval(sample_mode=sample_mode)
    eval_data = {key: value.float().to(device) for key, value in eval_data.items()}

    u = model(eval_data)["model_out"].detach().cpu().numpy().squeeze()
    u = 293.15 + 25 * u

    mesh = eval_data["coords"].detach().cpu().numpy().squeeze()
    beta = eval_data["beta"][0, :].detach().cpu().numpy()

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    sctt = ax.scatter3D(mesh[:, 0], mesh[:, 1], mesh[:, 2], c=u, cmap=cmap)
    fig.colorbar(
        sctt, ax=ax, shrink=0.5, aspect=5, ticks=np.linspace(u.min(), u.max(), 10)
    )
    plt.savefig(os.path.join(figure_dir, "eval_beta_{}.png".format(beta)))

plt.close("all")
