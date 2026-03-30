import time, torch, os, sys

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
sys.path.append("../../")
from src import dataio, training, loss_fun, modules

# ------------------ 基础层 domain_0 ------------------
domain_0 = dict(
    domain_name=0,
    geometry=dict(
        starts=[0.0, 0.0, 0.0],
        ends=[48, 48, 1.8],
        # 统一z轴网格间隔为3，与上层domain_1保持对齐
        num_intervals=[24, 24, 3],  # 修改：z轴间隔从9→3，生成4层网格点
        num_pde_points=25 * 25 * 4,     # 调整：对应新网格数 (24+1)*(24+1)*(3+1)=25 * 25 * 4
        num_single_bc_points=800,   # 调整：减少边界点
    ),
    conductivity_dist=dict(uneven_conductivity=False, background_conductivity=1),
    power=dict(
        bc=True,
        num_power_points_per_volume=4,
        num_power_points_per_surface=300,  # 调整：减少热源采样点
        num_power_points_per_cell=3,
        power_map=dict(
            power_0=dict(
                type="volumetric_power",
                location=dict(
                    starts=(0, 0, 0),    # 覆盖整个底层
                    ends=(24, 24, 3)     # 修改：z轴索引上限调整为3（原9）
                ),
                params=dict(k=0.2, value=1, weight=1),
            )
        ),
    ),
    # 边界条件调整：bottom改为自然对流方向
    front=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    back=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    left=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    right=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    bottom=dict(
        bc=True,
        type="htc",
        # 修改：direction=1表示向上散热（模拟自然对流）
        params=dict(dim=2, k=0.5, direction=1, weight=1)
    ),
    top=dict(bc=False),  # 由domain_1接管
    node=dict(root=True, leaf=False, children=dict(top=[2,3,4,5])),
    parameterized=dict(variable=False),
)

# ------------------ 中间层 domain_1 ------------------
domain_1 = dict(
    domain_name=1,
    geometry=dict(
        # 修改：完全覆盖domain_0顶部（x/y范围缩小但保证完全连接）
        starts=[9, 9, 1.8],    # 原9→12，确保热流连续
        ends=[39, 39, 1.95],      # 原39→36，高度压缩为0.2mm
        # 调整网格密度匹配上下层
        num_intervals=[15, 15, 3],  # 生成13x13x2网格点
        num_pde_points=1024,     # 调整：对应新网格数
        num_single_bc_points=500,   # 减少边界点
    ),
    # 修改：导电率调整到合理范围
    conductivity_dist=dict(uneven_conductivity=False, background_conductivity=1.0),
    power=dict(bc=False),
    front=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    back=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    left=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    right=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    bottom=dict(
        bc=True,
        type="htc",
        # 修改：direction=-1表示向下传导（对接domain_0的top）
        params=dict(dim=2, k=5.0, direction=-1, weight=1)  # 提高k值增强热耦合
    ),
    top=dict(bc=False),  # 由子域接管
    node=dict(
        root=False,
        leaf=False,
        # 修改：子域编号对应新的四象限划分
        children=dict(top=[2,3,4,5])
    ),
    parameterized=dict(variable=False),
)

# ------------------ 顶层子域 domain_2~5 ------------------
def create_top_domain(domain_id, x_start, x_end, y_start, y_end):
    return dict(
        domain_name=domain_id,
        geometry=dict(
            # 保持原有非覆盖设计：x/y范围在domain_1的[9-39]区间内
            starts=[x_start, y_start, 1.95],  # 示例：domain_2 starts=[11,11,1.95]
            ends=[x_end, y_end, 2.4],         # 示例：domain_2 ends=[23,23,2.4]
            # 保持原有网格设置
            num_intervals=[6, 6, 3],          # 生成7x7x4网格点
            num_pde_points=7 * 7 * 4,             # 196个PDE点
            num_single_bc_points=80
        ),
        conductivity_dist=dict(uneven_conductivity=False, background_conductivity=1),
        power=dict(bc=False),
        front=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
        back=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
        left=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
        right=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
        bottom=dict(
            bc=True,
            type="htc",
            # 参数严格匹配损失函数接口
            params=dict(
                dim=2,
                k=20.0,
                direction=-1,
                weight=1  # 仅保留必要参数
            )
        ),
        top=dict(
            bc=True,
            type="htc",
            params=dict(dim=2, k=0.5, direction=1, weight=1)
        ),
        node=dict(root=False, leaf=True, children={}),
        parameterized=dict(variable=False),
    )

# 保持原有非覆盖布局
domain_2 = create_top_domain(2, 11, 23, 11, 23)  # 左下 (x11-23, y11-23)
domain_3 = create_top_domain(3, 25, 37, 11, 23)  # 右下 (x25-37, y11-23)
domain_4 = create_top_domain(4, 11, 23, 25, 37)  # 左上 (x11-23, y25-37)
domain_5 = create_top_domain(5, 25, 37, 25, 37)  # 右上 (x25-37, y25-37)

domains_list = [domain_0, domain_1, domain_2, domain_3, domain_4, domain_5]
global_params = {
    "loss_fun_type": "norm",
    "num_params_per_epoch": 1,
    "pde_params": dict(type="pde", params=dict(k=0.2, weight=1)),
}

print(
    "Starting training single-case PINN: complex geometry and volumetric power defined in the middle layer"
)

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

dataset = dataio.CuboidGeometryDataIO(domains_list, global_params)
loss_fn = loss_fun.loss_fun_geometry_init(dataset)
val_func = training.val_fn_init(False)

root_path = "./log"
experiment_name = "experiment_5"
model_dir = os.path.join(root_path, experiment_name)
lr_decay = True
epochs_til_decay = 500
epochs_til_val = 200
epochs = 20
lr = 1e-3
epochs_til_checkpoints = 500

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
    deeponet=True,
)
toc = time.time()
print("total training time:", toc - tic)
