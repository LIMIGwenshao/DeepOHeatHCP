import time, torch, os, sys

# 获取当前文件路径并设置路径
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
sys.path.append("../")

# 引入所需的模块
from src import dataio_deeponet, training_deeponet_hcp, loss_fun_deeponet, modules

# 在这个例子中，domain 设置不完全正确，但是其结构仍然有参考价值。
domain_0 = dict(
    domain_name=0,
    geometry=dict(
        starts=[0.0, 0.0, 0.0],
        ends=[1.0, 1.0, 0.5],
        num_intervals=[20, 20, 10],
        num_pde_points=2000,
        num_single_bc_points=200,
    ),
    conductivity_dist=dict(uneven_conductivity=False, background_conductivity=1),
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
    front=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    back=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    left=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    right=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    bottom=dict(bc=True, type="htc", params=dict(dim=2, k=0.2, direction=-1, weight=1)),
    top=dict(bc=True),
    node=dict(root=True, leaf=True),
    parameterized=dict(variable=False),
)

# 设置全局参数
domains_list = [domain_0]
global_params = {
    "loss_fun_type": "norm",
    "num_params_per_epoch": 2,
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
    branch_in_features=441,
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

# 使用网格坐标加载数据
dataset = dataio_deeponet.DeepONetMeshDataIO(
    domains_list, global_params, dim=2, var=var, len_scale=len_scale
)

# 手动将数据集中的所有张量移到 GPU
# 遍历 dataset 的所有属性
for key, value in dataset.__dict__.items():  # 假设数据存储在 __dict__ 中
    if isinstance(value, torch.Tensor):  # 如果值是一个张量
        dataset.__dict__[key] = value.to(device)  # 将其移到 GPU

# 初始化损失函数（不需要迁移函数到 GPU）
loss_fn = loss_fun_deeponet.mesh_loss_fun_geometry_init(dataset)

# 初始化验证函数
val_func = training_deeponet_hcp.val_fn_init(False)

# 设置训练的路径和参数
root_path = "./log"
experiment_name = "testmodel"
model_dir = os.path.join(root_path, experiment_name)
lr_decay = True
epochs_til_decay = 500
epochs_til_val = 50
epochs = 10
lr = 1e-3
epochs_til_checkpoints = 1000

tic = time.time()

# 确保模型参数都在 GPU 上
for param in model.parameters():
    param.data = param.data.to(device)

# 开始训练
training_deeponet_hcp.train_mesh(
    model=model,
    dataset=dataset,
    epochs=epochs,
    lr=lr,
    epochs_til_checkpoints=epochs_til_checkpoints,
    model_dir=model_dir,
    loss_fn=loss_fn,  # 这里只传递函数，不需要迁移到 GPU
    val_fn=val_func,
    lr_decay=lr_decay,
    epochs_til_decay=epochs_til_decay,
    epochs_til_val=epochs_til_val,
    device=device,
)

toc = time.time()
print("total training time:", toc - tic)
