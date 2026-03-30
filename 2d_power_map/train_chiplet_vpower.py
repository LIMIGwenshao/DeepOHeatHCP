import time, torch, os, sys

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
sys.path.append("../../")
from src import dataio_deeponet, training_deeponet, loss_fun_deeponet, modules

# In this example the domain is not as functioning as before.
domain_0 = dict(
    domain_name=0,
    geometry=dict(
        starts=[0.0, 0.0, 0.0],#起始、终止以及间隔主要用于网格法建模
        ends=[1.0, 1.0, 0.5],
        num_intervals=[10, 10, 5],
        num_pde_points=2000,#这里是拉丁超采样的内部坐标点数量
        num_single_bc_points=200,#这里是拉丁超采样的边界点数量
    ),
    conductivity_dist=dict(uneven_conductivity=False, background_conductivity=1),
    power=dict(bc=False),
    front=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    back=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    left=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    right=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    #bottom=dict(bc=True),
    bottom=dict(bc=True, type="htc", params=dict(dim=2, k=0.2, direction=-1, weight=1)),
    top=dict(bc=True, type="adiabatics", params=dict(dim=2, weight=1)),
    #top=dict(bc=True, type="htc", params=dict(dim=2, k=0.2, direction=1, weight=1)),
    node=dict(root=True, leaf=False, children=dict(top=[1,2,3,4])),
    #node=dict(root=True, leaf=False, children=dict(top=[1])),
    parameterized=dict(variable=False),
)

domain_1 = dict(
    domain_name=1,
    geometry=dict(
        starts=[0.2, 0.2, 0.5],
        ends=[0.45, 0.45, 0.75],
        num_intervals=[5, 5, 2],
        num_pde_points=800,
        num_single_bc_points=200,
    ),
    conductivity_dist=dict(uneven_conductivity=False, background_conductivity=5),
    power=dict(bc=False),
    # power=dict(
    #     bc=True,
    #     num_power_points_per_volume=2,
    #     num_power_points_per_surface=500,
    #     num_power_points_per_cell=5,
    #     power_map=dict(
    #         power_0=dict(
    #             type="volumetric_power",
    #             location=dict(starts=(0, 0, 2), ends=(5, 5, 3)),
    #             params=dict(k=0.2, value=1, weight=1),
    #         )
    #     ),
    # ),
    front=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    back=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    left=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    right=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    bottom=dict(bc=False),
    #top=dict(bc=True),
    top=dict(bc=True, type="htc", params=dict(dim=2, k=0.2, direction=1, weight=1)),
    node=dict(root=False, leaf=True),
    parameterized=dict(variable=False),
)

domain_2 = dict(
    domain_name=2,
    geometry=dict(
        starts=[0.2, 0.55, 0.5],
        ends=[0.45, 0.8, 0.75],
        num_intervals=[5, 5, 2],
        num_pde_points=800,
        num_single_bc_points=200,
    ),
    conductivity_dist=dict(uneven_conductivity=False, background_conductivity=10),
    power=dict(bc=False),
    # power=dict(
    #     bc=True,
    #     num_power_points_per_volume=2,
    #     num_power_points_per_surface=500,
    #     num_power_points_per_cell=5,
    #     power_map=dict(
    #         power_0=dict(
    #             type="volumetric_power",
    #             location=dict(starts=(0, 0, 2), ends=(5, 5, 3)),
    #             params=dict(k=0.2, value=1, weight=1),
    #         )
    #     ),
    # ),
    front=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    back=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    left=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    right=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    bottom=dict(bc=False),
    #top=dict(bc=True),
    top=dict(bc=True, type="htc", params=dict(dim=2, k=0.2, direction=1, weight=1)),
    node=dict(root=False, leaf=True),
    parameterized=dict(variable=False),
)

domain_3 = dict(
    domain_name=3,
    geometry=dict(
        starts=[0.55, 0.2, 0.5],
        ends=[0.8, 0.45, 0.75],
        num_intervals=[5, 5, 2],
        num_pde_points=800,
        num_single_bc_points=200,
    ),
    conductivity_dist=dict(uneven_conductivity=False, background_conductivity=15),
    power=dict(bc=False),
    # power=dict(
    #     bc=True,
    #     num_power_points_per_volume=2,
    #     num_power_points_per_surface=500,
    #     num_power_points_per_cell=5,
    #     power_map=dict(
    #         power_0=dict(
    #             type="volumetric_power",
    #             location=dict(starts=(0, 0, 2), ends=(5, 5, 3)),
    #             params=dict(k=0.2, value=1, weight=1),
    #         )
    #     ),
    # ),
    front=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    back=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    left=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    right=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    bottom=dict(bc=False),
    #top=dict(bc=True),
    top=dict(bc=True, type="htc", params=dict(dim=2, k=0.2, direction=1, weight=1)),
    node=dict(root=False, leaf=True),
    parameterized=dict(variable=False),
)

domain_4 = dict(
    domain_name=4,
    geometry=dict(
        starts=[0.55, 0.55, 0.5],
        ends=[0.8, 0.8, 0.75],
        num_intervals=[5, 5, 2],
        num_pde_points=800,
        num_single_bc_points=200,
    ),
    conductivity_dist=dict(uneven_conductivity=False, background_conductivity=20),
    power=dict(
        bc=True,
        num_power_points_per_volume=2,
        num_power_points_per_surface=500,
        num_power_points_per_cell=5,
        power_map=dict(
            power_0=dict(
                type="surface_power",
                surface="top",
                location=dict(starts=(0, 0, 4), ends=(5, 5, 4)),
                params=dict(k=0.2, value=1, weight=1),
            )
        ),
    ),
    front=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    back=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    left=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    right=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    bottom=dict(bc=False),
    top=dict(bc=True),
    #top=dict(bc=True, type="htc", params=dict(dim=2, k=0.2, direction=1, weight=1)),
    node=dict(root=False, leaf=True),
    parameterized=dict(variable=False),
)

# Larger num_params_per_epoch is better. Here 50 is close to OOM on a single GPU.
#domains_list = [domain_0, domain_1]
domains_list = [domain_0, domain_1, domain_2, domain_3, domain_4]

global_params = {
    "loss_fun_type": "norm",#正则化,
    "num_params_per_epoch": 1,
    "pde_params": dict(type="pde", params=dict(k=0.2, weight=1)),
}

print("Starting training DeepOHeat: arbitrary 2D power map")

for i, domain in enumerate(domains_list):
    print("domain %d:" % i, domain)

device = "cuda:0"
model = modules.DeepONet(
    trunk_in_features=3,
    trunk_hidden_features=256,
    branch_in_features=121,
    branch_hidden_features=512,
    inner_prod_features=256,
    num_trunk_hidden_layers=3,
    num_branch_hidden_layers=7,
    nonlinearity="silu",
    freq=2 * torch.pi,
    std=1,
    freq_trainable=True,
    device=device,
)

print("The model used for this case:", model)

# The two variables control the functional space of the power map
# larger var indicates wider range of power value
# larger len_scale indicates smoother power distribution
var = 1 #这两个参数大概实际控制的是power map的高斯采样参数
len_scale = 0.3

print("var: {}, len scale: {}".format(var, len_scale))

# Here mesh coordinates are used. LHS design also works.
dataset = dataio_deeponet.DeepONetMeshDataIO( #数据集包括几何参数、各种参数、以及训练模式和测试模式、画图
    domains_list, global_params, dim=2, var=var, len_scale=len_scale
)#创造了一个数据集对象，将网格数据、参数化输入和其他相关信息封装，并转化为对象参数
loss_fn = loss_fun_deeponet.mesh_loss_fun_geometry_init(dataset) #loss是一个初始化的加权损失
val_func = training_deeponet.val_fn_init(False) #只用一半的左边，是切片（可能用于评估）

root_path = "./log"
experiment_name = "test517"
model_dir = os.path.join(root_path, experiment_name)
lr_decay = True
epochs_til_decay = 500
epochs_til_val = 50
epochs = 5000
lr = 1e-3
epochs_til_checkpoints = 1000

tic = time.time()
training_deeponet.train_mesh(
    model=model,
    dataset=dataset,
    epochs=epochs,
    lr=lr,
    epochs_til_checkpoints=epochs_til_checkpoints,
    model_dir=model_dir,
    loss_fn=loss_fn,
    val_fn=val_func,
    lr_decay=lr_decay,
    epochs_til_decay=epochs_til_decay,
    epochs_til_val=epochs_til_val,
    device=device,
)
toc = time.time()
print("total training time:", toc - tic)

