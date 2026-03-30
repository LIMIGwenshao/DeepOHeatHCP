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

        eval_data = dataset.eval() #生成验证数据

        if device is not None:
            eval_data = {key: value.to(device) for key, value in eval_data.items()}

        eval_data = {key: value.float() for key, value in eval_data.items()}
        model.eval()#模型验证

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
            model_input, power_map, conductivity, geometry_step = dataset.train()

            if device is not None:
                model_input = {
                    key: value.to(device) for key, value in model_input.items()
                }
                conductivity = conductivity.to(device)
                power_map = power_map.to(device)

            model_input = {key: value.float() for key, value in model_input.items()}
            coords_list.append(model_input["coords"])
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
    model, #模型
    dataset, #数据集
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

    def generate_loss_dicts(): #生成损失值字典的函数
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

        return loss_dict, loss_idx_count_dict #返回的两个一模一样

    total_loss_list = []
    pde_loss_list = []
    htc_loss_list = []
    adiabatics_loss_list = []
    volumetric_powers_loss_list = []
    surface_powers_loss_list = []
    neumanns_loss_list = []
    dirichelets_loss_list = []


    epoch = 0 #迭代次数初始化
    # dataset.draw_power_map(model_dir) #画power map，这里不需要
    for epoch in range(start_epoch, epochs): #迭代循环
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

        loss_dict, _ = generate_loss_dicts() #在这里实际生成损失值字典
        starting_idx = 0
        coords_list, beta_list = [], []
        conductivity_list, geometry_list, idx_list = [], [], []

        for step in range(len(dataset)):
            model_input, conductivity, geometry_step = dataset.train() #生成训练数据：包括模型输入（字典数据结构），热导率，几何

            if device is not None:
                model_input = {
                    key: value.to(device) for key, value in model_input.items()
                } # 将model_input中的所有张量转移到指定设备
                conductivity = conductivity.to(device) #转移设备操作

            model_input = {key: value.float() for key, value in model_input.items()}##转移设备操作改变数据格式
            coords_list.append(model_input["coords"]) #坐标数据输入
            beta_list.append(model_input["beta"]) #训练数据输入

            ending_idx = int(starting_idx + (model_input["coords"]).shape[0])
            idx_list.append((starting_idx, ending_idx))
            starting_idx = ending_idx #更新起始索引

            conductivity_list.append(conductivity)
            geometry_list.append(geometry_step)

        model_input = dict(
            coords=torch.concat(coords_list, 0), beta=torch.concat(beta_list, 0)
        )
        conductivity = torch.concat(conductivity_list, 0)#torch.concat 是 PyTorch 中用于将多个张量沿指定维度拼接的函数
        beta = model_input["beta"]
        coords = model_input["coords"]
        # print(f"Beta.shape: {beta.shape}")
        # print(f"Beta: {beta}")
        # print(f"Coords.shape: {coords.shape}")
        # print(f"Coords: {coords}")

        model_output = model(model_input)
        u = model_output["model_out"]
        coords = model_output["model_in"]

        jac, u_laplace = laplacian_jacobian(u, coords, conductivity) # 计算梯度


        for step in range(len(dataset)):
            idx_step = idx_list[step]
            start_idx, end_idx = idx_step[0], idx_step[1]

            geometry_step = geometry_list[step] #在每个几何里逐步迭代
            loss_dict_step = loss_fn(
                u[start_idx:end_idx, ...], #预测温度值
                jac[start_idx:end_idx, ...],#梯度
                u_laplace[start_idx:end_idx, ...],#梯度
                beta[start_idx:end_idx, ...],#模型输入，计算体积功率把这里注释掉
                geometry_step,#传入当前几何
            )#看一下返回的是什么 return top_2d_power_loss_fn #返回的是这个函数

            if loss_dict.keys() != loss_dict_step.keys():
                raise ValueError(
                    f"Loss dict in current step is not aligned with the pre-defined loss dict, unable to merge"
                )
            else:
                for key in loss_dict.keys():
                    loss_dict[key] += loss_dict_step[key]#这里是将各项损失值计算的核心步骤

        loss_dict["total_loss"] = sum([*loss_dict.values()]) / len(dataset) #平均误差损失

        print("epoch: %d" % epoch, loss_dict)

        optim.zero_grad()
        loss_dict["total_loss"].backward() #反向传播
        optim.step()

        #将各项损失值存储起来
        total_loss_list.append(loss_dict["total_loss"].item())
        pde_loss_list.append(loss_dict["pde"].item())
        htc_loss_list.append(loss_dict["htc"].item())
        adiabatics_loss_list.append(loss_dict["adiabatics"].item())
        #volumetric_powers_loss_list.append(loss_dict["volumetric_power"].item())
        surface_powers_loss_list.append(loss_dict["surface_power"].item())
        #neumanns_loss_list.append(loss_dict["neumanns"].item())
        #dirichelets_loss_list.append(loss_dict["dirichelets"].item())


        # Save the loss lists as .npy files
        np.save('DeepONet_total_loss_list.npy', total_loss_list)
        np.save('DeepONet_pde_loss_list.npy', pde_loss_list)
        np.save('DeepONet_htc_loss_list.npy', htc_loss_list)
        np.save('DeepONet_adiabatics_loss_list.npy', adiabatics_loss_list)
        #np.save('DeepONet_volumetric_power_loss_list.npy', volumetric_powers_loss_list)
        np.save('DeepONet_surface_power_loss_list.npy', surface_powers_loss_list)
        #np.save('DeepONet_neumanns_loss_list.npy', neumanns_loss_list)
        #np.save('DeepONet_dirichlets_loss_list.npy', dirichelets_loss_list)



    #画出各项损失值下降曲线
    plt.figure(figsize=(16, 8))
    plt.plot(np.arange(len(total_loss_list)), total_loss_list)
    #plt.yscale("linear")
    plt.yscale("log")
    plt.savefig(os.path.join(figure_dir, "loss.png"))
    plt.close("all")

    plt.figure(figsize=(16, 8))
    plt.plot(np.arange(len(pde_loss_list)), pde_loss_list)
    # plt.yscale("linear")
    plt.yscale("log")
    plt.savefig(os.path.join(figure_dir, "pde_loss.png"))
    plt.close("all")

    plt.figure(figsize=(16, 8))
    plt.plot(np.arange(len(htc_loss_list)), htc_loss_list)
    # plt.yscale("linear")
    plt.yscale("log")
    plt.savefig(os.path.join(figure_dir, "htc_loss.png"))
    plt.close("all")

    plt.figure(figsize=(16, 8))
    plt.plot(np.arange(len(adiabatics_loss_list)), adiabatics_loss_list)
    # plt.yscale("linear")
    plt.yscale("log")
    plt.savefig(os.path.join(figure_dir, "adiabatics_loss.png"))
    plt.close("all")

    # plt.figure(figsize=(16, 8))
    # plt.plot(np.arange(len(volumetric_powers_loss_list)), volumetric_powers_loss_list)
    # # plt.yscale("linear")
    # plt.yscale("log")
    # plt.savefig(os.path.join(figure_dir, "volumetric_loss.png"))
    # plt.close("all")

    plt.figure(figsize=(16, 8))
    plt.plot(np.arange(len(surface_powers_loss_list)), surface_powers_loss_list)
    # plt.yscale("linear")
    plt.yscale("log")
    plt.savefig(os.path.join(figure_dir, "surface_power_loss.png"))
    plt.close("all")

    # plt.figure(figsize=(16, 8))
    # plt.plot(np.arange(len(neumanns_loss_list)), neumanns_loss_list)
    # # plt.yscale("linear")
    # plt.yscale("log")
    # plt.savefig(os.path.join(figure_dir, "neumanns_loss.png"))
    # plt.close("all")
    #
    # plt.figure(figsize=(16, 8))
    # plt.plot(np.arange(len(dirichelets_loss_list)), dirichelets_loss_list)
    # # plt.yscale("linear")
    # plt.yscale("log")
    # plt.savefig(os.path.join(figure_dir, "dirichelets_loss.png"))
    # plt.close("all")


    val_fn(dataset, model, epoch + 1, figure_dir, device) #这一步是训练结束后最后验证，不是训练过程中的验证

    checkpoint = {
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
    }
    torch.save(
        checkpoint, os.path.join(checkpoint_dir, "model_epoch_{}.pth".format(epoch + 1))
    )
    print(f"successfully saved checkpoint at epoch: {epoch + 1}")
