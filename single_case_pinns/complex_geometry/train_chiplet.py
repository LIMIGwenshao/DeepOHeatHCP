import time, torch, os, sys

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
sys.path.append("../../")
from src import dataio, training_hcp, loss_fun, modules

domain_0 = dict(
    domain_name=0,
    geometry=dict(
        starts=[0.0, 0.0, 0.0],
        ends=[2.0, 2.0, 0.55],
        num_intervals=[20, 20, 10],
        num_pde_points=4000, #不知道怎么确定的
        num_single_bc_points=500, #不知道怎么确定的
    ),
    conductivity_dist=dict(uneven_conductivity=False, background_conductivity=10),
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
        num_intervals=[10, 10, 4],
        num_pde_points=600,
        num_single_bc_points=100,
    ),
    conductivity_dist=dict(uneven_conductivity=False, background_conductivity=5),
    power=dict(bc=False),
    front=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    back=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    left=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    right=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    bottom=dict(bc=True, type="interface", params=dict(dim=2, weight=1)),
    top=dict(bc=True, type="htc", params=dict(dim=2, k=0.2, direction=1, weight=1)),
    node=dict(root=False, leaf=True),
    parameterized=dict(variable=False),
)

domain_2 = dict(
    domain_name=2,
    geometry=dict(
        starts=[0.4, 1.1, 0.55],
        ends=[0.9, 1.6, 0.75],
        num_intervals=[10, 10, 4],
        num_pde_points=600,
        num_single_bc_points=100,
    ),
    conductivity_dist=dict(uneven_conductivity=False, background_conductivity=10),
    power=dict(bc=False),
    front=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    back=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    left=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    right=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    bottom=dict(bc=True, type="interface", params=dict(dim=2, weight=1)),
    top=dict(bc=True, type="htc", params=dict(dim=2, k=0.2, direction=1, weight=1)),
    node=dict(root=False, leaf=True),
    parameterized=dict(variable=False),
)

domain_3 = dict(
    domain_name=3,
    geometry=dict(
        starts=[1.1, 0.4, 0.55],
        ends=[1.6, 0.9, 0.75],
        num_intervals=[10, 10, 4],
        num_pde_points=600,
        num_single_bc_points=100,
    ),
    conductivity_dist=dict(uneven_conductivity=False, background_conductivity=15),
    power=dict(bc=False),
    front=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    back=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    left=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    right=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    bottom=dict(bc=True, type="interface", params=dict(dim=2, weight=1)),
    top=dict(bc=True, type="htc", params=dict(dim=2, k=0.2, direction=1, weight=1)),
    node=dict(root=False, leaf=True),
    parameterized=dict(variable=False),
)

domain_4 = dict(
    domain_name=4,
    geometry=dict(
        starts=[1.1, 1.1, 0.55],
        ends=[1.6, 1.6, 0.75],
        num_intervals=[10, 10, 4],
        num_pde_points=600,
        num_single_bc_points=100,
    ),
    conductivity_dist=dict(uneven_conductivity=False, background_conductivity=20),
    power=dict(bc=False),
    front=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    back=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    left=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    right=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    bottom=dict(bc=True, type="interface", params=dict(dim=2, weight=1)),
    top=dict(bc=True, type="htc", params=dict(dim=2, k=0.2, direction=1, weight=1)),
    node=dict(root=False, leaf=True),
    parameterized=dict(variable=False),
)
domains_list = [domain_0, domain_1, domain_2, domain_3, domain_4]

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
print("The dataset used for this case:", dataset)
loss_fn = loss_fun.loss_fun_geometry_init(dataset)
val_func = training_hcp.val_fn_init(False)

root_path = "./log"
experiment_name = "10000_mesh+lhs_chiplet"
model_dir = os.path.join(root_path, experiment_name)
lr_decay = True
epochs_til_decay = 500
epochs_til_val = 200
epochs = 10000
lr = 1e-3
epochs_til_checkpoints = 500

tic = time.time()
training_hcp.train(
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





