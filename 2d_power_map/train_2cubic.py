import time, torch, os, sys

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
sys.path.append("../../")
from src import dataio_mesh, training, loss_fun, modules

# In this example the domain is not as functioning as before.
domain_0 = dict(
    domain_name=0,
    geometry=dict(
        starts=[0.0, 0.0, 0.0],
        ends=[2.0, 1.0, 0.55],
        num_intervals=[20, 10, 11],
        num_pde_points=4000,
        num_single_bc_points=500,
    ),
    conductivity_dist=dict(uneven_conductivity=False, background_conductivity=1),
    power=dict(
        bc=True,
        num_power_points_per_volume=4,
        num_power_points_per_surface=500,
        num_power_points_per_cell=5,
        power_map=dict(
            power_0=dict(
                type="volumetric_power",
                location=dict(starts=(0, 0, 5), ends=(20, 10, 6)),
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
    node=dict(root=True, leaf=False, children=dict(top=[1])),
    parameterized=dict(variable=False),
)

domain_1 = dict(
    domain_name=1,
    geometry=dict(
        starts=[0.0, 0.0, 0.55],
        ends=[1.0, 1.0, 0.75],
        num_intervals=[10, 10, 4],
        num_pde_points=800,
        num_single_bc_points=200,
    ),
    conductivity_dist=dict(uneven_conductivity=False, background_conductivity=1),
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

# Larger num_params_per_epoch is better. Here 50 is close to OOM on a single GPU.
#domains_list = [domain_0, domain_1]
domains_list = [domain_0, domain_1]

global_params = {
    "loss_fun_type": "norm",#正则化,
    "num_params_per_epoch": 1,
    "pde_params": dict(type="pde", params=dict(k=0.2, weight=1)),
}

print("Starting training DeepOHeat: arbitrary 2D power map")

for i, domain in enumerate(domains_list):
    print("domain %d:" % i, domain)

device = "cuda:0"
model = modules.FFN(
    nonlinearity="silu",
    in_features=3,
    num_hidden_layers=3,
    hidden_features=128,
    device=device,
    freq=torch.pi,
    freq_trainable=True,
)
print("The model used for this case:", model)

# The two variables control the functional space of the power map
# larger var indicates wider range of power value
# larger len_scale indicates smoother power distribution
var = 1 #这两个参数大概实际控制的是power map的高斯采样参数
len_scale = 0.3

print("var: {}, len scale: {}".format(var, len_scale))

# Here mesh coordinates are used. LHS design also works.
dataset = dataio_mesh.CuboidGeometryDataIO( #数据集包括几何参数、各种参数、以及训练模式和测试模式、画图
    domains_list, global_params, dim=2, var=var, len_scale=len_scale
)#创造了一个数据集对象，将网格数据、参数化输入和其他相关信息封装，并转化为对象参数

loss_fn = loss_fun.loss_fun_geometry_init(dataset) #loss是一个初始化的加权损失
val_func = training.val_fn_init(False) #只用一半的左边，是切片（可能用于评估）

root_path = "./log"
experiment_name = "test"
model_dir = os.path.join(root_path, experiment_name)
lr_decay = True
epochs_til_decay = 200
epochs_til_val = 20
epochs = 100
lr = 2e-3
epochs_til_checkpoints = 1000

tic = time.time()
training.train(
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





