import shutil
from statistics import mode
import torch
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from src.diff_operator import laplacian_jacobian
from src.utils import MyCmap


def val_fn_init(half_geometry=False, slice_dim=None, slice_value=None):
    if half_geometry and slice_dim is None and slice_value is None:
        raise ValueError(f"No slice coordinate is given")
    cmap = MyCmap.get_cmap()

    def val_fn(dataset, model, epoch, figure_dir, device=None):

        eval_data = dataset.eval()

        if device is not None:
            eval_data = {key: value.to(device) for key, value in eval_data.items()}

        eval_data = {key: value.float() for key, value in eval_data.items()}
        model.eval()

        u = model(eval_data)["model_out"].detach().cpu().numpy().squeeze()
        u = 293.15 + 25 * u

        mesh = eval_data["coords"].detach().cpu().numpy().squeeze()

        if half_geometry:
            slice_idx = mesh[:, slice_dim] >= slice_value
            mesh = mesh[slice_idx, :]
            u = u[slice_idx]

        fig = plt.figure()
        ax = plt.axes(projection="3d")
        sctt = ax.scatter3D(mesh[:, 0], mesh[:, 1], mesh[:, 2], c=u, cmap=cmap)
        fig.colorbar(
            sctt, ax=ax, shrink=0.5, aspect=5, ticks=np.linspace(u.min(), u.max(), 10)
        )
        plt.savefig(os.path.join(figure_dir, "pred_epoch_{}.png".format(epoch)))
        plt.close("all")

    return val_fn


def train(
    model,
    dataset,
    epochs,
    lr,
    epochs_til_checkpoints,
    model_dir,
    loss_fn,
    val_fn=None,
    device=None,
    start_epoch=0,
    lr_decay=False,
    epochs_til_decay=100,
    epochs_til_val=500,
):

    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    lr_decayed = lr

    if start_epoch > 0:
        model_path = os.path.join(
            model_dir, "checkpoints", "model_epoch_{}.pth".format(start_epoch)
        )
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model"])
        model.train()
        optim.load_state_dict(checkpoint["optim"])
        optim.param_groups[0]["lr"] = lr
        # assert(start_epoch == checkpoint['epoch'])

    else:
        if os.path.exists(model_dir):
            val = input("Path %s already exists, overwrite? (y/n)" % model_dir)
            if val == "y":
                shutil.rmtree(model_dir)
        os.makedirs(model_dir)

    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    figure_dir = os.path.join(model_dir, "figure")
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    def generate_loss_dicts():
        loss_dict = {
            "pde": 0,
            "htc": 0,
            "adiabatics": 0,
            "volumetric_power": 0,
            "surface_power": 0,
            "neumann": 0,
            "dirichelet": 0,
        }
        loss_idx_count_dict = loss_dict.copy()

        return loss_dict, loss_idx_count_dict

    total_loss_list = []
    epoch = 0
    dataset.draw_power_map(model_dir)
    for epoch in range(start_epoch, epochs):
        if val_fn and not epoch % epochs_til_val and epoch:
            val_fn(dataset, model, epoch, figure_dir, device)

        if not epoch % epochs_til_checkpoints and epoch:
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optim.state_dict(),
            }
            torch.save(
                checkpoint,
                os.path.join(checkpoint_dir, "model_epoch_{}.pth".format(epoch)),
            )

        if not epoch % epochs_til_decay and epoch and lr_decay:
            lr_decayed = lr_decayed * 0.9
            optim = torch.optim.Adam(lr=lr_decayed, params=model.parameters())
            print("lr decay, current: {}".format(lr_decayed))

        loss_dict, _ = generate_loss_dicts()
        starting_idx = 0
        coords_list, beta_list = [], []
        power_map_list, conductivity_list, geometry_list, idx_list = [], [], [], []

        for step in range(len(dataset)):
            model_input, power_map, conductivity, geometry_step = dataset.train() #geometry是几何根结点的cubic

            if device is not None:
                model_input = {
                    key: value.to(device) for key, value in model_input.items()
                }
                conductivity = conductivity.to(device)
                power_map = power_map.to(device)

            model_input = {key: value.float() for key, value in model_input.items()} #数据类型转换
            coords_list.append(model_input["coords"]) #将数据导入list
            beta_list.append(model_input["beta"])

            ending_idx = int(starting_idx + (model_input["coords"]).shape[0])
            idx_list.append((starting_idx, ending_idx))
            starting_idx = ending_idx

            power_map_list.append(power_map)
            conductivity_list.append(conductivity)
            geometry_list.append(geometry_step)

        model_input = dict(
            coords=torch.concat(coords_list, 0), beta=torch.concat(beta_list, 0)
        )
        model_output = model(model_input)

        u = model_output["model_out"]
        coords = model_output["model_in"]

        power_map = torch.concat(power_map_list, 0)
        conductivity = torch.concat(conductivity_list, 0)

        jac, u_laplace = laplacian_jacobian(u, coords, conductivity)

        for step in range(len(dataset)):
            idx_step = idx_list[step]
            start_idx, end_idx = idx_step[0], idx_step[1]

            geometry_step = geometry_list[step]
            loss_dict_step = loss_fn(
                u[start_idx:end_idx, ...],
                jac[start_idx:end_idx, ...],
                u_laplace[start_idx:end_idx, ...],
                power_map[start_idx:end_idx],
                geometry_step,
            )

            if loss_dict.keys() != loss_dict_step.keys():
                raise ValueError(
                    f"Loss dict in current step is not aligned with the pre-defined loss dict, unable to merge"
                )
            else:
                for key in loss_dict.keys():
                    loss_dict[key] += loss_dict_step[key]

        loss_dict["total_loss"] = sum([*loss_dict.values()]) / len(dataset)

        print("epoch: %d" % epoch, loss_dict)

        optim.zero_grad()
        loss_dict["total_loss"].backward()
        optim.step()

        total_loss_list.append(loss_dict["total_loss"].item())

    plt.figure(figsize=(16, 8))
    plt.plot(np.arange(len(total_loss_list)), total_loss_list)
    plt.yscale("log")
    plt.savefig(os.path.join(figure_dir, "loss.png"))
    plt.close("all")

    val_fn(dataset, model, epoch + 1, figure_dir, device)

    checkpoint = {
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
    }
    torch.save(
        checkpoint, os.path.join(checkpoint_dir, "model_epoch_{}.pth".format(epoch + 1))
    )


def train_mesh(
    model,
    dataset,
    epochs,
    lr,
    epochs_til_checkpoints,
    model_dir,
    loss_fn,
    val_fn=None,
    device=None,
    start_epoch=0,
    lr_decay=False,
    epochs_til_decay=100,
    epochs_til_val=500,
):

    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    lr_decayed = lr

    # Move model to device (GPU or CPU)
    if device is not None:
        model = model.to(device)

    if start_epoch > 0:
        model_path = os.path.join(
            model_dir, "checkpoints", "model_epoch_{}.pth".format(start_epoch)
        )
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model"])
        model.train()
        optim.load_state_dict(checkpoint["optim"])
        optim.param_groups[0]["lr"] = lr
        assert start_epoch == checkpoint["epoch"]

    else:
        if os.path.exists(model_dir):
            val = input("Path %s already exists, overwrite? (y/n)" % model_dir)
            if val == "y":
                shutil.rmtree(model_dir)
        os.makedirs(model_dir)

    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    figure_dir = os.path.join(model_dir, "figure")
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    def generate_loss_dicts():
        loss_dict = {
            "pde": 0,
            "htc": 0,
            "adiabatics": 0,
            "volumetric_power": 0,
            "surface_power": 0,
            "neumann": 0,
            "dirichelet": 0,
        }
        loss_idx_count_dict = loss_dict.copy()

        return loss_dict, loss_idx_count_dict

    total_loss_list = []
    pde_loss_list = []
    htc_loss_list = []
    adiabatics_loss_list = []
    volumetric_powers_loss_list = []
    surface_powers_loss_list = []
    neumanns_loss_list = []
    dirichelets_loss_list = []



    epoch = 0
    dataset.draw_power_map(model_dir)
    for epoch in range(start_epoch, epochs):
        if val_fn and not epoch % epochs_til_val and epoch:
            val_fn(dataset, model, epoch, figure_dir, device)

        if not epoch % epochs_til_checkpoints and epoch:
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optim.state_dict(),
            }
            torch.save(
                checkpoint,
                os.path.join(checkpoint_dir, "model_epoch_{}.pth".format(epoch)),
            )

        if not epoch % epochs_til_decay and epoch and lr_decay:
            lr_decayed = lr_decayed * 0.9
            optim = torch.optim.Adam(lr=lr_decayed, params=model.parameters())
            print("lr decay, current: {}".format(lr_decayed))

        loss_dict, _ = generate_loss_dicts()
        starting_idx = 0
        coords_list, beta_list = [], []
        conductivity_list, geometry_list, idx_list = [], [], []

        # Loop over the dataset and move all inputs to the device (GPU)
        for step in range(len(dataset)):
            model_input, conductivity, geometry_step = dataset.train()

            if device is not None:
                model_input = {key: value.to(device) for key, value in model_input.items()}
                conductivity = conductivity.to(device)

            model_input = {key: value.float() for key, value in model_input.items()}
            coords_list.append(model_input["coords"])
            beta_list.append(model_input["beta"])

            ending_idx = int(starting_idx + (model_input["coords"]).shape[0])
            idx_list.append((starting_idx, ending_idx))
            starting_idx = ending_idx

            conductivity_list.append(conductivity)
            geometry_list.append(geometry_step)

        model_input = dict(
            coords=torch.concat(coords_list, 0).to(device),
            beta=torch.concat(beta_list, 0).to(device)
        )
        conductivity = torch.concat(conductivity_list, 0).to(device)
        beta = model_input["beta"]

        model_output = model(model_input)  # Output

        u_initial = model_output["model_out"]  # u is the output
        coords = model_output["model_in"]

        # Build the constraint matrix A
        A = build_constraint_matrix_batch(coords, 21, 11, 11, 1e-3 / 20, 1e-3 / 20,
                                          0.5e-3 / 10, 0.1).to(device)

        # Extract local predictions (including center and neighboring points)
        y_pred_local = extract_local_predictions_batch(coords, u_initial, 21, 21, 11).to(device)
        y_pred_local.requires_grad_(True)

        # 选择1%的点
        batch_size = coords.size(0)
        num_points = coords.size(1)  # 假设 coords 的形状是 (batch_size, num_points, 3)
        num_active_points = int(num_points * 0.01)  # 1%的点

        # 随机选择1%的点的索引
        random_indices = torch.randint(0, num_points, (batch_size, num_active_points), device=device)

        # 扁平化 u_initial 和 u_adjusted
        u_initial_flattened = u_initial.view(-1)  # [batch_size * num_points]
        useven = projection_batch(A, y_pred_local)
        u_adjusted = useven[:, 0].unsqueeze(-1)  # 保持调整后的形状为 [batch_size, num_points, 1]
        u_adjusted_flattened = u_adjusted.view(-1)  # [batch_size * num_points]

        # 创建一个新的 u，初始化为 u_initial
        u = u_initial.clone()

        # 对每个批次，更新1%的点
        for i in range(batch_size):
            # 选择当前批次的随机索引
            selected_indices = random_indices[i]
            # 计算对应的扁平化索引
            flat_indices = selected_indices + i * num_points  # 扁平化的索引

            # 使用更新后的 u_adjusted 进行替换
            u_initial_flattened[flat_indices] = u_adjusted_flattened[flat_indices]

        # 将更新后的 u 还原为原始的形状
        u = u_initial_flattened.view(u.shape)  # 将 u 还原回原来的形状

        jac, u_laplace = laplacian_jacobian(u, coords, conductivity)

        for step in range(len(dataset)):
            idx_step = idx_list[step]
            start_idx, end_idx = idx_step[0], idx_step[1]

            geometry_step = geometry_list[step]
            loss_dict_step = loss_fn(
                u[start_idx:end_idx, ...],
                jac[start_idx:end_idx, ...],
                u_laplace[start_idx:end_idx, ...],
                beta[start_idx:end_idx, ...],
                geometry_step,
            )

            if loss_dict.keys() != loss_dict_step.keys():
                raise ValueError(
                    f"Loss dict in current step is not aligned with the pre-defined loss dict, unable to merge"
                )
            else:
                for key in loss_dict.keys():
                    loss_dict[key] += loss_dict_step[key]

        loss_dict["total_loss"] = sum([*loss_dict.values()]) / len(dataset)

        print("epoch: %d" % epoch, loss_dict)

        optim.zero_grad()
        loss_dict["total_loss"].backward()
        optim.step()

        # 将各项损失值存储起来
        total_loss_list.append(loss_dict["total_loss"].item())
        pde_loss_list.append(loss_dict["pde_loss"].item())
        htc_loss_list.append(loss_dict["htc_loss"].item())
        adiabatics_loss_list.append(loss_dict["adiabatics_loss"].item())
        volumetric_powers_loss_list.append(loss_dict["volumetric_powers"].item())
        surface_powers_loss_list.append(loss_dict["surface_powers"].item())
        neumanns_loss_list.append(loss_dict["neumanns_loss"].item())
        dirichelets_loss_list.append(loss_dict["dirichelets"].item())

        # Save the loss lists as .npy files
        np.save('HCP_total_loss_list.npy', total_loss_list)
        np.save('HCP_pde_loss_list.npy', pde_loss_list)
        np.save('HCP_htc_loss_list.npy', htc_loss_list)
        np.save('HCP_adiabatics_loss_list.npy', adiabatics_loss_list)
        np.save('HCP_volumetric_power_loss_list.npy', volumetric_powers_loss_list)
        np.save('HCP_surface_power_loss_list.npy', surface_powers_loss_list)
        np.save('HCP_neumanns_loss_list.npy', neumanns_loss_list)
        np.save('HCP_dirichlets_loss_list.npy', dirichelets_loss_list)



    #画出各项损失值下降曲线
    plt.figure(figsize=(16, 8))
    plt.plot(np.arange(len(total_loss_list)), total_loss_list)
    #plt.yscale("linear")
    plt.yscale("log")
    plt.savefig(os.path.join(figure_dir, "HCP_loss.png"))
    plt.close("all")

    plt.figure(figsize=(16, 8))
    plt.plot(np.arange(len(pde_loss_list)), pde_loss_list)
    # plt.yscale("linear")
    plt.yscale("log")
    plt.savefig(os.path.join(figure_dir, "HCP_pde_loss.png"))
    plt.close("all")

    plt.figure(figsize=(16, 8))
    plt.plot(np.arange(len(htc_loss_list)), htc_loss_list)
    # plt.yscale("linear")
    plt.yscale("log")
    plt.savefig(os.path.join(figure_dir, "HCP_htc_loss.png"))
    plt.close("all")

    plt.figure(figsize=(16, 8))
    plt.plot(np.arange(len(adiabatics_loss_list)), adiabatics_loss_list)
    # plt.yscale("linear")
    plt.yscale("log")
    plt.savefig(os.path.join(figure_dir, "HCP_adiabatics_loss.png"))
    plt.close("all")

    plt.figure(figsize=(16, 8))
    plt.plot(np.arange(len(volumetric_powers_loss_list)), volumetric_powers_loss_list)
    # plt.yscale("linear")
    plt.yscale("log")
    plt.savefig(os.path.join(figure_dir, "HCP_volumetric_loss.png"))
    plt.close("all")

    plt.figure(figsize=(16, 8))
    plt.plot(np.arange(len(surface_powers_loss_list)), surface_powers_loss_list)
    # plt.yscale("linear")
    plt.yscale("log")
    plt.savefig(os.path.join(figure_dir, "HCP_surface_power_loss.png"))
    plt.close("all")

    plt.figure(figsize=(16, 8))
    plt.plot(np.arange(len(neumanns_loss_list)), neumanns_loss_list)
    # plt.yscale("linear")
    plt.yscale("log")
    plt.savefig(os.path.join(figure_dir, "HCP_neumanns_loss.png"))
    plt.close("all")

    plt.figure(figsize=(16, 8))
    plt.plot(np.arange(len(dirichelets_loss_list)), dirichelets_loss_list)
    # plt.yscale("linear")
    plt.yscale("log")
    plt.savefig(os.path.join(figure_dir, "HCP_dirichelets_loss.png"))
    plt.close("all")


    val_fn(dataset, model, epoch + 1, figure_dir, device)

    checkpoint = {
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
    }
    torch.save(
        checkpoint, os.path.join(checkpoint_dir, "model_epoch_{}.pth".format(epoch + 1))
    )


def build_constraint_matrix_batch(x, Nx, Ny, Nz, dx, dy, dz, k):
    batch_size = x.size(0)
    A = torch.zeros((batch_size, 7), device=x.device)  # Ensure A is on the correct device

    A[:, 0] = -2.0 * k * (1.0 / dx ** 2 + 1.0 / dy ** 2 + 1.0 / dz ** 2)  # Center point
    A[:, 1] = k / dx ** 2  # Left point
    A[:, 2] = k / dx ** 2  # Right point
    A[:, 3] = k / dy ** 2  # Bottom point
    A[:, 4] = k / dy ** 2  # Top point
    A[:, 5] = k / dz ** 2  # Front point
    A[:, 6] = k / dz ** 2  # Back point

    return A.unsqueeze(1)


def projection_batch(A, H):
    I = torch.eye(A.size()[2], device=A.device).float().repeat(A.size()[0], 1, 1)  # Ensure I is on the correct device

    invs = torch.inverse(torch.bmm(A, A.permute(0, 2, 1)))

    intermediate = torch.bmm(torch.bmm(A.permute(0, 2, 1), invs), A)
    H = H.unsqueeze(-1)

    H_ = torch.bmm((I - intermediate), H)

    return H_.squeeze(2)


def extract_local_predictions_batch(x, y_pred, Nx, Ny, Nz):
    batch_size = x.size(0)

    # 创建一个空的张量来存储局部预测值
    local_predictions = torch.zeros((batch_size, 7), device=x.device)

    # 计算每个batch中的i, j, k索引
    i_indices = torch.arange(batch_size, device=x.device) // (Ny * Nz)  # i（高度）维度的索引
    j_indices = (torch.arange(batch_size, device=x.device) // Nz) % Ny  # j（宽度）维度的索引
    k_indices = torch.arange(batch_size, device=x.device) % Nz  # k（深度）维度的索引

    # 计算局部预测值的索引
    idx_center = i_indices * Ny * Nz + j_indices * Nz + k_indices
    idx_left = torch.maximum(i_indices - 1, torch.zeros_like(i_indices, device=x.device)) * Ny * Nz + j_indices * Nz + k_indices
    idx_right = torch.minimum(i_indices + 1, torch.tensor(Nx - 1, device=x.device)) * Ny * Nz + j_indices * Nz + k_indices
    idx_down = i_indices * Ny * Nz + torch.maximum(j_indices - 1, torch.zeros_like(j_indices, device=x.device)) * Nz + k_indices
    idx_up = i_indices * Ny * Nz + torch.minimum(j_indices + 1, torch.tensor(Ny - 1, device=x.device)) * Nz + k_indices
    idx_front = i_indices * Ny * Nz + j_indices * Nz + torch.maximum(k_indices - 1, torch.zeros_like(k_indices, device=x.device))
    idx_back = i_indices * Ny * Nz + j_indices * Nz + torch.minimum(k_indices + 1, torch.tensor(Nz - 1, device=x.device))

    # 确保 y_pred 是一维张量，并通过 index 访问
    y_pred = y_pred.view(-1)  # 确保 y_pred 是一维张量

    # 使用计算出的索引填充局部预测值张量
    local_predictions[:, 0] = y_pred[idx_center]
    local_predictions[:, 1] = y_pred[idx_left]
    local_predictions[:, 2] = y_pred[idx_right]
    local_predictions[:, 3] = y_pred[idx_down]
    local_predictions[:, 4] = y_pred[idx_up]
    local_predictions[:, 5] = y_pred[idx_front]
    local_predictions[:, 6] = y_pred[idx_back]

    return local_predictions
