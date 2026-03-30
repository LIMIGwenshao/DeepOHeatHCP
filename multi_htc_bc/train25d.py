import time, torch, os, sys

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
sys.path.append("../../")
from src import dataio_deeponet, training_deeponet, loss_fun_deeponet, modules


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
    domain_name=1,
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
    domain_name=1,
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
    domain_name=1,
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
    "Starting training DeepOHeat: parameterized top HTC BC with volumetric power defined in the middle layer"
)

for i, domain in enumerate(domains_list):
    print("domain %d:" % i, domain)

device = "cuda:0"
model = modules.HierarchicalDeepONet(
    domains_list,
    trunk_in_features=3,
    trunk_hidden_features=128,
    branch_hidden_features=128,
    inner_prod_features=64,
    num_trunk_hidden_layers=3,
    num_branch_hidden_layers=3,
    nonlinearity="silu",
    freq=2 * torch.pi,
    std=1,
    freq_trainable=True,
    device=device,
)
print("The model used for this case:", model)

# print("The model used for this case:", model)

# The two variables control the functional space of the power map
# larger var indicates wider range of power value
# larger len_scale indicates smoother power distribution
var = 1 #这两个参数大概实际控制的是power map的高斯采样参数
len_scale = 0.3

print("var: {}, len scale: {}".format(var, len_scale))

# Here mesh coordinates are used. LHS design also works.
dataset = dataio_deeponet.DeepONetMeshDataIO( #数据集包括几何参数、各种参数、以及训练模式和测试模式、画图
    domains_list, global_params, dim=3, var=var, len_scale=len_scale #dim=3
)#创造了一个数据集对象，将网格数据、参数化输入和其他相关信息封装，并转化为对象参数
sample = dataset.train()

loss_fn = loss_fun_deeponet.mesh_loss_fun_geometry_init(dataset) #loss是一个初始化的加权损失
val_func = training_deeponet.val_fn_init(False) #只用一半的左边，是切片（可能用于评估）

root_path = "./log"
experiment_name = "experiment_test2"
model_dir = os.path.join(root_path, experiment_name)
lr_decay = True
epochs_til_decay = 500
epochs_til_val = 50
epochs = 50
lr = 1e-3
epochs_til_checkpoints = 200

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
    #deeponet=True,
)
toc = time.time()
print("total training time:", toc - tic) #ddd

