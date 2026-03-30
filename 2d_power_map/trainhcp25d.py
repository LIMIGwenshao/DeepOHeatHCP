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
        num_pde_points=600, #不知道怎么确定的
        num_single_bc_points=200, #不知道怎么确定的
    ),
    conductivity_dist=dict(uneven_conductivity=False, background_conductivity=1),
    # Define this power is only to do eval during training. This power is not used in training.
    power=dict(
        bc=True,
        num_power_points_per_volume=2,
        num_power_points_per_surface=500,
        num_power_points_per_cell=5,
        power_map=dict(
            power_0=dict(
                type="surface_power",
                surface="top",
                location=dict(starts=(10, 0, 10), ends=(20, 10, 10)),
                params=dict(dim=2, value=1, weight=1),
            )
        ),
    ),
# domain_0 = dict(
#     domain_name=0,
#     geometry=dict(
#         starts=[0.0, 0.0, 0.0],
#         ends=[2.0, 2.0, 0.55],
#         num_intervals=[20, 20, 11],
#         num_pde_points=600, #不知道怎么确定的
#         num_single_bc_points=200, #不知道怎么确定的
#     ),
#     conductivity_dist=dict(uneven_conductivity=False, background_conductivity=5),
#     power=dict(
#         bc=True,
#         num_power_points_per_volume=4,
#         num_power_points_per_surface=500,
#         num_power_points_per_cell=5,
#         power_map=dict(
#             power_0=dict(
#                 type="volumetric_power",
#                 location=dict(starts=(0, 0, 5), ends=(20, 20, 6)),
#                 params=dict(k=0.2, value=1, weight=1),
#             )
#         ),
#     ),
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
        num_pde_points=100,
        num_single_bc_points=20,
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
        num_pde_points=100,
        num_single_bc_points=20,
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
        num_pde_points=100,
        num_single_bc_points=20,
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
        num_pde_points=100,
        num_single_bc_points=20,
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
# 设置全局参数
domains_list = [domain_0, domain_1, domain_2, domain_3, domain_4]
global_params = {
    "loss_fun_type": "norm",
    "num_params_per_epoch": 1,
    "pde_params": dict(type="pde", params=dict(k=0.2, weight=1)),
}

print("Starting training DeepOHeat: arbitrary 2D power map")

# 打印当前 domain 信息
for i, domain in enumerate(domains_list):
    print("domain %d:" % i, domain)

# 设置设备为 GPU
device = "cuda:0"  # Make sure everything is moved to the GPU

# 创建 DeepONet 模型
model = modules.DeepONet(
    trunk_in_features=3,
    trunk_hidden_features=128,
    branch_in_features=585,
    branch_hidden_features=256,
    inner_prod_features=128,
    num_trunk_hidden_layers=3,
    num_branch_hidden_layers=7,
    nonlinearity="silu",
    freq=2 * torch.pi,
    std=1,
    freq_trainable=True,
    device=device,
)
print("The model used for this case:", model)

# 设置控制 power map 的变量

var = 1
len_scale = 0.3

print("var: {}, len scale: {}".format(var, len_scale))

# 创建数据集（修改为3D）
dataset = dataio_deeponet.DeepONetMeshDataIO(
    domains_list, global_params, dim=2, var=var, len_scale=len_scale
)

# # GPU迁移函数
# def move_to_gpu(dataset, device):
#     for key, value in dataset.__dict__.items():
#         if isinstance(value, torch.Tensor):
#             dataset.__dict__[key] = value.to(device)
#         elif isinstance(value, dict):
#             for k, v in value.items():
#                 if isinstance(v, torch.Tensor):
#                     value[k] = v.to(device)
#     return dataset

# 迁移数据到GPU
# dataset = move_to_gpu(dataset, device)

# 初始化损失函数和验证函数
loss_fn = loss_fun_deeponet.mesh_loss_fun_geometry_init(dataset)
val_func = training_deeponet.val_fn_init(False)

# 设置训练的路径和参数
root_path = "./log"
experiment_name = "5000_25d_mesh_SurfacePower"
model_dir = os.path.join(root_path, experiment_name)
lr_decay = True
epochs_til_decay = 500
epochs_til_val = 50
epochs = 10000
lr = 1e-3
epochs_til_checkpoints = 1000

tic = time.time()

# 确保模型参数都在 GPU 上
for param in model.parameters():
    param.data = param.data.to(device)

# 开始训练
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
