import torch
import torch.nn as nn
from collections import OrderedDict


class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


def xavier_init(layer):
    with torch.no_grad():
        if type(layer) == nn.Linear:
            if hasattr(layer, "weight"):
                nn.init.xavier_normal_(layer.weight)
        else:
            raise TypeError(f"Expecting nn.Linear got type={type(layer)} instead")


class FCBlock(nn.Module):

    def __init__(
        self,
        out_features=1,
        in_features=3,
        hidden_features=20,
        num_hidden_layers=3,
        nonlinearity="sine",
        device=None,
    ):
        super().__init__()

        nl_init_dict = dict(
            sine=(Sine(), xavier_init),
            silu=(nn.SiLU(), xavier_init),
            tanh=(nn.Tanh(), xavier_init),
            relu=(nn.ReLU(), xavier_init),
        )
        nl, init = nl_init_dict[nonlinearity]

        self.net = OrderedDict()

        for i in range(num_hidden_layers + 2):
            if i == 0:
                self.net["fc1"] = nn.Linear(
                    in_features=in_features, out_features=hidden_features
                )
                self.net["nl1"] = nl
            elif i != num_hidden_layers + 1:
                self.net["fc%d" % (i + 1)] = nn.Linear(
                    in_features=hidden_features, out_features=hidden_features
                )
                self.net["nl%d" % (i + 1)] = nl

            else:
                self.net["fc%d" % (i + 1)] = nn.Linear(
                    in_features=hidden_features, out_features=out_features
                )

            init(self.net["fc%d" % (i + 1)])

        self.net = nn.Sequential(self.net)

        if device:
            self.net.to(device)

    def forward(self, x):
        return self.net(x)


class DNN(nn.Module):

    def __init__(
        self,
        out_features=1,
        in_features=3,
        hidden_features=20,
        num_hidden_layers=3,
        nonlinearity="sine",
        device=None,
    ):
        super().__init__()

        self.net = FCBlock(
            out_features=out_features,
            in_features=in_features,
            hidden_features=hidden_features,
            num_hidden_layers=num_hidden_layers,
            nonlinearity=nonlinearity,
            device=device,
        )

    def forward(self, model_input):
        coords_org = model_input["coords"].clone().detach().requires_grad_(True)
        coords = coords_org

        output = self.net(coords)
        return {"model_in": coords_org, "model_out": output}


class FFN(nn.Module):#神经网络的训练是针对与单个点的，也就是说输入有多少个点就有多少个神经网络

    def __init__(
        self,
        out_features=1,
        in_features=3,
        hidden_features=20,
        num_hidden_layers=3,
        nonlinearity="sine",
        device=None,
        freq=torch.pi,
        std=1,
        freq_trainable=True,
    ):#这里只定义参数
        super().__init__()

        self.net = FCBlock(
            out_features=out_features,
            in_features=hidden_features,
            hidden_features=hidden_features,
            num_hidden_layers=num_hidden_layers,
            nonlinearity=nonlinearity,
            device=device,
        )#这一步是网络

        self.freq = nn.Parameter(
            torch.ones(1, device=device) * freq, requires_grad=freq_trainable
        )

        self.fourier_features = nn.Parameter(
            torch.zeros(in_features, int(hidden_features / 2), device=device).normal_(
                0, std
            )
            * self.freq,
            requires_grad=False,
        )

    def forward(self, model_input):
        coords_org = model_input["coords"].clone().detach().requires_grad_(True)
        coords = coords_org

        ff_input = torch.concat(
            [
                torch.sin(torch.matmul(coords, self.fourier_features)),
                torch.cos(torch.matmul(coords, self.fourier_features)),
            ],
            -1,
        )

        output = self.net(ff_input)
        return {"model_in": coords_org, "model_out": output}


class ModifiedFC(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        nonlinearity="sine",
        transform=True,
        activate=True,
    ):
        super().__init__()

        nl_init_dict = dict(
            sine=(Sine(), xavier_init),
            silu=(nn.SiLU(), xavier_init),
            tanh=(nn.Tanh(), xavier_init),
            relu=(nn.ReLU(), xavier_init),
        )

        self.nl, self.init_fun = nl_init_dict[nonlinearity]
        self.transform = transform
        self.activate = activate

        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.init_fun(self.fc)

    def forward(self, x, trans_1, trans_2):

        if self.transform:
            output = self.nl(self.fc(x))
            return (1 - output) * trans_1 + output * trans_2

        elif self.activate:
            return self.nl(self.fc(x))

        else:
            return self.fc(x)


class ModifiedFCBlock(nn.Module):

    def __init__(
        self,
        out_features=1,
        in_features=3,
        hidden_features=20,
        num_hidden_layers=3,
        nonlinearity="sine",
        device=None,
    ):
        super().__init__()

        nl_init_dict = dict(
            sine=(Sine(), xavier_init),
            silu=(nn.SiLU(), xavier_init),
            tanh=(nn.Tanh(), xavier_init),
            relu=(nn.ReLU(), xavier_init),
        )
        nl, init = nl_init_dict[nonlinearity]

        self.net = []

        for i in range(num_hidden_layers + 2):
            if i == 0:
                self.net.append(
                    ModifiedFC(
                        in_features=in_features,
                        out_features=hidden_features,
                        nonlinearity=nonlinearity,
                        transform=False,
                        activate=True,
                    )
                )
            elif i != num_hidden_layers + 1:
                self.net.append(
                    ModifiedFC(
                        in_features=hidden_features,
                        out_features=hidden_features,
                        nonlinearity=nonlinearity,
                        transform=True,
                    )
                )
            else:
                self.net.append(
                    ModifiedFC(
                        in_features=hidden_features,
                        out_features=out_features,
                        transform=False,
                        activate=False,
                    )
                )

        self.transform_layer_1 = nn.Linear(
            in_features=in_features, out_features=hidden_features
        )
        init(self.transform_layer_1)
        self.transform_layer_2 = nn.Linear(
            in_features=in_features, out_features=hidden_features
        )
        init(self.transform_layer_2)

        self.net = nn.ModuleList(self.net)

        if device:
            self.net.to(device)
            self.transform_layer_1.to(device)
            self.transform_layer_2.to(device)

    def forward(self, x):
        trans_1 = self.transform_layer_1(x)
        trans_2 = self.transform_layer_2(x)

        for net_i in self.net:
            x = net_i(x, trans_1, trans_2)
        return x


class ModifiedFCBlockFourier(nn.Module):

    def __init__(
        self,
        out_features=1,
        in_features=3,
        hidden_features=20,
        num_hidden_layers=3,
        nonlinearity="sine",
        device=None,
    ):
        super().__init__()

        nl_init_dict = dict(
            sine=(Sine(), xavier_init),
            silu=(nn.SiLU(), xavier_init),
            tanh=(nn.Tanh(), xavier_init),
            relu=(nn.ReLU(), xavier_init),
        )
        nl, init = nl_init_dict[nonlinearity]

        self.net = []

        for i in range(num_hidden_layers + 2):
            if i == 0:
                self.net.append(
                    ModifiedFC(
                        in_features=in_features,
                        out_features=hidden_features,
                        nonlinearity=nonlinearity,
                        transform=False,
                        activate=True,
                    )
                )
            elif i != num_hidden_layers + 1:
                self.net.append(
                    ModifiedFC(
                        in_features=hidden_features,
                        out_features=hidden_features,
                        nonlinearity=nonlinearity,
                        transform=True,
                    )
                )
            else:
                self.net.append(
                    ModifiedFC(
                        in_features=hidden_features,
                        out_features=out_features,
                        transform=False,
                        activate=False,
                    )
                )

        self.transform_layer_1 = nn.Linear(
            in_features=hidden_features, out_features=hidden_features
        )
        init(self.transform_layer_1)
        self.transform_layer_2 = nn.Linear(
            in_features=hidden_features, out_features=hidden_features
        )
        init(self.transform_layer_2)

        self.net = nn.ModuleList(self.net)

        if device:
            self.net.to(device)
            self.transform_layer_1.to(device)
            self.transform_layer_2.to(device)

    def forward(self, x, fourier_features):
        trans_1 = self.transform_layer_1(fourier_features)
        trans_2 = self.transform_layer_2(fourier_features)

        for net_i in self.net:
            x = net_i(x, trans_1, trans_2)
        return x


class ModifiedDNN(nn.Module):

    def __init__(
        self,
        out_features=1,
        in_features=3,
        hidden_features=20,
        num_hidden_layers=3,
        nonlinearity="sine",
        device=None,
    ):
        super().__init__()

        self.net = ModifiedFCBlock(
            out_features=out_features,
            in_features=in_features,
            hidden_features=hidden_features,
            num_hidden_layers=num_hidden_layers,
            nonlinearity=nonlinearity,
            device=device,
        )

    def forward(self, model_input):
        coords_org = model_input["coords"].clone().detach().requires_grad_(True)
        coords = coords_org

        output = self.net(coords)
        return {"model_in": coords_org, "model_out": output}


class ModifiedFFN(nn.Module):

    def __init__(
        self,
        out_features=1,
        in_features=3,
        hidden_features=20,
        num_hidden_layers=3,
        nonlinearity="sine",
        device=None,
        freq=torch.pi,
        std=1,
        freq_trainable=True,
    ):
        super().__init__()

        self.net = ModifiedFCBlockFourier(
            out_features=out_features,
            in_features=in_features,
            hidden_features=hidden_features,
            num_hidden_layers=num_hidden_layers,
            nonlinearity=nonlinearity,
            device=device,
        )

        self.freq = nn.Parameter(
            torch.ones(1, device=device) * torch.pi, requires_grad=freq_trainable
        )
        self.fourier_features = nn.Parameter(
            torch.zeros(in_features, int(hidden_features / 2), device=device).normal_(
                0, std
            )
            * self.freq,
            requires_grad=False,
        )

    def forward(self, model_input):
        coords_org = model_input["coords"].clone().detach().requires_grad_(True)
        coords = coords_org

        ff_input = torch.concat(
            [
                torch.sin(torch.matmul(coords, self.fourier_features)),
                torch.cos(torch.matmul(coords, self.fourier_features)),
            ],
            -1,
        )

        output = self.net(coords, ff_input)
        return {"model_in": coords_org, "model_out": output}


class DeepONet(nn.Module):

    def __init__(
        self,
        trunk_in_features=3,
        trunk_hidden_features=128,
        branch_in_features=1,
        branch_hidden_features=20,
        inner_prod_features=50,
        num_branch_hidden_layers=3,
        num_trunk_hidden_layers=3,
        nonlinearity="silu",
        freq=torch.pi,
        std=1,
        freq_trainable=True,
        device=None,
    ):
        super().__init__()

        self.branch = FCBlock(
            out_features=inner_prod_features,
            in_features=branch_in_features,
            hidden_features=branch_hidden_features,
            num_hidden_layers=num_branch_hidden_layers,
            nonlinearity=nonlinearity,
            device=device,
        )

        self.trunk = FCBlock(
            out_features=inner_prod_features,
            in_features=trunk_hidden_features,
            hidden_features=trunk_hidden_features,
            num_hidden_layers=num_trunk_hidden_layers,
            nonlinearity=nonlinearity,
            device=device,
        )

        self.freq = nn.Parameter(
            torch.ones(1, device=device) * freq, requires_grad=freq_trainable
        )
        self.fourier_features = nn.Parameter(
            torch.zeros(
                trunk_in_features, int(trunk_hidden_features / 2), device=device
            ).normal_(0, std)
            * self.freq,
            requires_grad=False,
        )
        self.b_0 = nn.Parameter(
            torch.zeros(1, device=device).uniform_(), requires_grad=True
        )

    def forward(self, model_input):
        coords_org = model_input["coords"].clone().detach().requires_grad_(True)
        coords = coords_org

        beta = model_input["beta"].clone().detach().requires_grad_(True)
        # print("beta shape:", beta.shape)
        # #print("beta values:", beta)
        # print("coords shape:", coords.shape)

        ff_input = torch.concat(
            [
                torch.sin(torch.matmul(coords, self.fourier_features)),
                torch.cos(torch.matmul(coords, self.fourier_features)),
            ],
            -1,
        )

        output_1 = self.trunk(ff_input)
        output_2 = self.branch(beta)

        output = torch.sum(output_1 * output_2, 1).reshape(-1, 1) + self.b_0
        return {"model_in": coords_org, "model_out": output}


class BranchNetList(nn.Module):

    def __init__(self, net_arc, num_nets, *args):
        super().__init__()

        self.net_list = nn.ModuleList([net_arc(*args) for i in range(num_nets)])

    def forward(self, x, trunk_output):
        output = trunk_output
        for i, branch_i in enumerate(self.net_list):
            output *= branch_i(x[:, i].reshape(-1, 1))

        return output


class MIONet(nn.Module):

    def __init__(
        self,
        trunk_in_features=3,
        trunk_hidden_features=128,
        branch_in_features=1,
        branch_hidden_features=20,
        inner_prod_features=50,
        num_hidden_layers=3,
        nonlinearity="silu",
        freq=torch.pi,
        std=1,
        freq_trainable=True,
        device=None,
    ):
        super().__init__()

        self.branch = BranchNetList(
            FCBlock,
            branch_in_features,
            inner_prod_features,
            1,
            branch_hidden_features,
            num_hidden_layers,
            nonlinearity,
            device,
        )

        self.trunk = FCBlock(
            out_features=inner_prod_features,
            in_features=trunk_hidden_features,
            hidden_features=trunk_hidden_features,
            num_hidden_layers=num_hidden_layers,
            nonlinearity=nonlinearity,
            device=device,
        )

        self.freq = nn.Parameter(
            torch.ones(1, device=device) * freq, requires_grad=freq_trainable
        )
        self.fourier_features = nn.Parameter(
            torch.zeros(
                trunk_in_features, int(trunk_hidden_features / 2), device=device
            ).normal_(0, std)
            * self.freq,
            requires_grad=False,
        )
        self.b_0 = nn.Parameter(
            torch.zeros(1, device=device).uniform_(), requires_grad=True
        )

    def forward(self, model_input):
        coords_org = model_input["coords"].clone().detach().requires_grad_(True)
        coords = coords_org

        beta = model_input["beta"].clone().detach().requires_grad_(True)
        # print("beta shape:", beta.shape)
        # print("beta values:", beta)

        ff_input = torch.concat(
            [
                torch.sin(torch.matmul(coords, self.fourier_features)),
                torch.cos(torch.matmul(coords, self.fourier_features)),
            ],
            -1,
        )

        output = (
            torch.sum(self.branch(beta, self.trunk(ff_input)), 1).reshape(-1, 1)
            + self.b_0
        )

        return {"model_in": coords_org, "model_out": output}


class FFONet(nn.Module):

    def __init__(
        self,
        trunk_in_features=3,
        trunk_hidden_features=128,
        branch_in_features=1,
        branch_hidden_features=20,
        inner_prod_features=50,
        num_branch_hidden_layers=3,
        num_trunk_hidden_layers=3,
        nonlinearity="silu",
        trunk_freq=torch.pi,
        branch_freq=torch.pi,
        std=1,
        freq_trainable=True,
        device=None,
    ):
        super().__init__()

        self.branch = FCBlock(
            out_features=inner_prod_features,
            in_features=branch_hidden_features,
            hidden_features=branch_hidden_features,
            num_hidden_layers=num_branch_hidden_layers,
            nonlinearity=nonlinearity,
            device=device,
        )

        self.trunk = FCBlock(
            out_features=inner_prod_features,
            in_features=trunk_hidden_features,
            hidden_features=trunk_hidden_features,
            num_hidden_layers=num_trunk_hidden_layers,
            nonlinearity=nonlinearity,
            device=device,
        )

        self.trunk_freq = nn.Parameter(
            torch.ones(1, device=device) * trunk_freq, requires_grad=freq_trainable
        )
        self.branch_freq = nn.Parameter(
            torch.ones(1, device=device) * branch_freq, requires_grad=freq_trainable
        )

        self.branch_fourier_features = nn.Parameter(
            torch.zeros(
                branch_in_features, int(branch_hidden_features / 2), device=device
            ).normal_(0, std)
            * self.branch_freq,
            requires_grad=False,
        )
        self.trunk_fourier_features = nn.Parameter(
            torch.zeros(
                trunk_in_features, int(trunk_hidden_features / 2), device=device
            ).normal_(0, std)
            * self.trunk_freq,
            requires_grad=False,
        )
        self.b_0 = nn.Parameter(
            torch.zeros(1, device=device).uniform_(), requires_grad=True
        )

    def forward(self, model_input):
        coords_org = model_input["coords"].clone().detach().requires_grad_(True)
        coords = coords_org

        beta = model_input["beta"].clone().detach().requires_grad_(True)

        trunk_ff_input = torch.concat(
            [
                torch.sin(torch.matmul(coords, self.trunk_fourier_features)),
                torch.cos(torch.matmul(coords, self.trunk_fourier_features)),
            ],
            -1,
        )
        branch_ff_input = torch.concat(
            [
                torch.sin(torch.matmul(beta, self.branch_fourier_features)),
                torch.cos(torch.matmul(beta, self.branch_fourier_features)),
            ],
            -1,
        )

        output_1 = self.trunk(trunk_ff_input)
        output_2 = self.branch(branch_ff_input)

        output = torch.sum(output_1 * output_2, 1).reshape(-1, 1) + self.b_0
        return {"model_in": coords_org, "model_out": output}

#
# class HierarchicalDeepONet(nn.Module):
#     def __init__(
#         self,
#         domains_list,
#         trunk_in_features=3,
#         trunk_hidden_features=128,
#         branch_hidden_features=128,
#         inner_prod_features=64,
#         num_trunk_hidden_layers=3,
#         num_branch_hidden_layers=3,
#         nonlinearity="silu",
#         freq=2 * torch.pi,
#         std=1,
#         freq_trainable=True,
#         device=None,
#     ):
#         super().__init__()
#
#         # 保存设备信息
#         self.device = device
#
#         # 保存域信息
#         self.domains = domains_list
#
#         # 计算每个域的网格点数
#         self.domain_sizes = {}
#         total_points = 0
#
#         # 打印domains_list的内容以便调试
#         print("Domains list:", domains_list)
#
#         for i, domain in enumerate(domains_list):
#             # 使用简单的数字索引作为域名称
#             domain_name = f"domain_{i}"
#             # 计算该域的网格点数
#             intervals = domain['geometry']['num_intervals']
#             num_points = (intervals[0] + 1) * (intervals[1] + 1) * (intervals[2] + 1)
#
#             # 保存域大小信息
#             self.domain_sizes[domain_name] = num_points
#             total_points += num_points
#
#             # 打印调试信息
#             print(f"Domain {domain_name} has {num_points} points")
#
#         print(f"Total points across all domains: {total_points}")
#
#         # 为每种边界条件创建分支网络
#         self.bc_nets = nn.ModuleDict({
#             'htc': FCBlock(
#                 out_features=inner_prod_features,
#                 in_features=1,  # 热传导系数
#                 hidden_features=branch_hidden_features // 4,
#                 num_hidden_layers=num_branch_hidden_layers // 2,
#                 nonlinearity=nonlinearity,
#                 device=device,
#             ),
#             'adiabatics': FCBlock(
#                 out_features=inner_prod_features,
#                 in_features=1,  # 绝热边界
#                 hidden_features=branch_hidden_features // 4,
#                 num_hidden_layers=num_branch_hidden_layers // 2,
#                 nonlinearity=nonlinearity,
#                 device=device,
#             ),
#             'volumetric_power': FCBlock(
#                 out_features=inner_prod_features,
#                 in_features=1,  # 体积热源
#                 hidden_features=branch_hidden_features // 4,
#                 num_hidden_layers=num_branch_hidden_layers // 2,
#                 nonlinearity=nonlinearity,
#                 device=device,
#             )
#         })
#
#         # 主干网络
#         self.trunk = FCBlock(
#             out_features=inner_prod_features,
#             in_features=trunk_hidden_features,
#             hidden_features=trunk_hidden_features,
#             num_hidden_layers=num_trunk_hidden_layers,
#             nonlinearity=nonlinearity,
#             device=device,
#         )
#
#         # 傅里叶特征
#         self.freq = nn.Parameter(
#             torch.ones(1, device=device) * freq,
#             requires_grad=freq_trainable
#         )
#         self.fourier_features = nn.Parameter(
#             torch.zeros(
#                 trunk_in_features, int(trunk_hidden_features / 2), device=device
#             ).normal_(0, std)
#             * self.freq,
#             requires_grad=False,
#         )
#
#         # 偏置项
#         self.b_0 = nn.Parameter(
#             torch.zeros(1, device=device).uniform_(),
#             requires_grad=True
#         )
#
#         # 边界条件权重
#         self.bc_weights = nn.Parameter(
#             torch.ones(len(self.bc_nets), device=device),
#             requires_grad=True
#         )
#
#         # 将整个模型移动到指定设备
#         if device is not None:
#             self.to(device)
#
#     def forward(self, model_input):
#         # 添加输入验证
#         assert "coords" in model_input, "Missing coordinates in input"
#         assert "beta" in model_input, "Missing beta in input"
#
#         # 使用torch.no_grad()减少不必要的梯度计算
#         with torch.set_grad_enabled(self.training):
#             coords_org = model_input["coords"].clone().detach().requires_grad_(True)
#             coords = coords_org
#             beta = model_input["beta"].clone().detach().requires_grad_(True)
#
#             # 确保所有张量都在同一设备上
#             if self.device is not None:
#                 coords = coords.to(self.device)
#                 beta = beta.to(self.device)
#
#             # 打印输入数据的形状
#             print(f"Input shapes - coords: {coords.shape}, beta: {beta.shape}")
#
#             # 傅里叶特征变换
#             ff_input = torch.concat(
#                 [
#                     torch.sin(torch.matmul(coords, self.fourier_features)),
#                     torch.cos(torch.matmul(coords, self.fourier_features)),
#                 ],
#                 -1,
#             )
#
#             # 主干网络处理
#             trunk_output = self.trunk(ff_input)
#
#             # 处理边界条件
#             bc_outputs = []
#
#             # 为每个边界条件创建一个默认值
#             default_bc_value = torch.ones(beta.shape[0], 1, device=beta.device)
#
#             for bc_name, bc_net in self.bc_nets.items():
#                 # 使用默认值作为边界条件输入
#                 bc_data = default_bc_value
#                 print(f"Processing boundary condition: {bc_name}")
#                 print(f"BC data shape: {bc_data.shape}")
#
#                 bc_output = bc_net(bc_data)
#                 bc_outputs.append(bc_output)
#
#             # 组合所有输出
#             bc_sum = torch.zeros_like(bc_outputs[0])
#             for i, output in enumerate(bc_outputs):
#                 bc_sum += output * self.bc_weights[i].view(1, 1)
#
#             # 最终输出
#             output = torch.sum(trunk_output * bc_sum, 1).reshape(-1, 1) + self.b_0
#
#             return {"model_in": coords_org, "model_out": output} #dd
#
#
# class DomainBasedMIONet(nn.Module):
#     def __init__(
#         self,
#         domains_list,
#         trunk_in_features=3,
#         trunk_hidden_features=128,
#         branch_hidden_features=20,
#         inner_prod_features=50,
#         num_hidden_layers=3,
#         nonlinearity="silu",
#         freq=torch.pi,
#         std=1,
#         freq_trainable=True,
#         device=None,
#     ):
#         super().__init__()
#
#         # 保存设备信息
#         self.device = device
#
#         # 保存域信息
#         self.domains = domains_list
#
#         # 计算每个域的网格点数
#         self.domain_sizes = {}
#         total_points = 0
#
#         for i, domain in enumerate(domains_list):
#             # 使用域名称作为键
#             domain_name = domain.get('domain_name', f"domain_{i}")
#
#             # 计算该域的网格点数
#             geometry = domain.get('geometry', {})
#             num_intervals = geometry.get('num_intervals', [1, 1, 1])
#             num_pde_points = geometry.get('num_pde_points', 0)
#             num_single_bc_points = geometry.get('num_single_bc_points', 0)
#
#             # 计算边界点数
#             bc_points = 0
#             for bc_key in ['front', 'back', 'left', 'right', 'bottom', 'top']:
#                 if bc_key in domain and domain[bc_key].get('bc', False):
#                     bc_points += num_single_bc_points
#
#             # 总点数 = PDE点数 + 边界点数
#             total_domain_points = num_pde_points + bc_points
#
#             # 保存域大小信息
#             self.domain_sizes[domain_name] = total_domain_points
#             total_points += total_domain_points
#
#             print(f"Domain {domain_name} has {total_domain_points} points")
#
#         print(f"Total points across all domains: {total_points}")
#
#         # 主干网络 - 处理所有坐标输入
#         self.trunk = FCBlock(
#             out_features=inner_prod_features,
#             in_features=trunk_hidden_features,
#             hidden_features=trunk_hidden_features,
#             num_hidden_layers=num_hidden_layers,
#             nonlinearity=nonlinearity,
#             device=device,
#         )
#
#         # 为每个边界创建分支网络
#         self.boundary_nets = nn.ModuleDict()
#         self.boundary_sizes = {}
#         total_boundaries = 0
#
#         for i, domain in enumerate(domains_list):
#             domain_name = domain.get('domain_name', f"domain_{i}")
#
#             # 检查每个边界
#             for bc_key in ['front', 'back', 'left', 'right', 'bottom', 'top']:
#                 if bc_key in domain and domain[bc_key].get('bc', False):
#                     # 创建边界网络名称
#                     boundary_name = f"{domain_name}_{bc_key}"
#
#                     # 获取边界点数
#                     num_single_bc_points = domain.get('geometry', {}).get('num_single_bc_points', 0)
#
#                     # 保存边界大小信息
#                     self.boundary_sizes[boundary_name] = num_single_bc_points
#                     total_boundaries += 1
#
#                     # 创建边界网络
#                     self.boundary_nets[boundary_name] = FCBlock(
#                         out_features=inner_prod_features,
#                         in_features=1,  # 每个边界只使用一个参数
#                         hidden_features=branch_hidden_features,
#                         num_hidden_layers=num_hidden_layers,
#                         nonlinearity=nonlinearity,
#                         device=device,
#                     )
#
#                     print(f"Created boundary network for {boundary_name} with {num_single_bc_points} points")
#
#         print(f"Total boundaries: {total_boundaries}")
#
#         # 傅里叶特征
#         self.freq = nn.Parameter(
#             torch.ones(1, device=device) * freq,
#             requires_grad=freq_trainable
#         )
#         self.fourier_features = nn.Parameter(
#             torch.zeros(
#                 trunk_in_features, int(trunk_hidden_features / 2), device=device
#             ).normal_(0, std) * self.freq,
#             requires_grad=False,
#         )
#
#         # 偏置项
#         self.b_0 = nn.Parameter(
#             torch.zeros(1, device=device).uniform_(),
#             requires_grad=True
#         )
#
#         # 边界权重
#         self.boundary_weights = nn.Parameter(
#             torch.ones(total_boundaries, device=device),
#             requires_grad=True
#         )
#
#         # 将整个模型移动到指定设备
#         if device is not None:
#             self.to(device)
#
#     def forward(self, model_input):
#         # 获取输入
#         coords_org = model_input["coords"].clone().detach().requires_grad_(True)
#         coords = coords_org
#         beta = model_input["beta"].clone().detach().requires_grad_(True)
#
#         # 确保所有张量都在同一设备上
#         if self.device is not None:
#             coords = coords.to(self.device)
#             beta = beta.to(self.device)
#             # 将整个模型移动到设备上
#             self.to(self.device)
#
#         # 打印调试信息
#         # print("beta shape:", beta.shape)
#         # print("beta values:", beta)
#         # print("coords device:", coords.device)
#         # print("fourier_features device:", self.fourier_features.device)
#
#         # 傅里叶特征变换
#         ff_input = torch.concat(
#             [
#                 torch.sin(torch.matmul(coords, self.fourier_features)),
#                 torch.cos(torch.matmul(coords, self.fourier_features)),
#             ],
#             -1,
#         )
#
#         # 主干网络处理
#         trunk_output = self.trunk(ff_input)
#
#         # 处理每个边界的输出
#         boundary_outputs = []
#         current_idx = 0
#
#         for i, domain in enumerate(self.domains):
#             domain_name = domain.get('domain_name', f"domain_{i}")
#
#             # 检查每个边界
#             for bc_key in ['front', 'back', 'left', 'right', 'bottom', 'top']:
#                 if bc_key in domain and domain[bc_key].get('bc', False):
#                     # 创建边界网络名称
#                     boundary_name = f"{domain_name}_{bc_key}"
#
#                     # 获取边界大小
#                     boundary_size = self.boundary_sizes[boundary_name]
#
#                     # 提取对应边界的数据
#                     boundary_data = beta[:, current_idx:current_idx + boundary_size]
#
#                     # 取平均值作为边界参数
#                     boundary_param = boundary_data.mean(dim=1, keepdim=True)
#
#                     # 处理边界数据
#                     boundary_output = self.boundary_nets[boundary_name](boundary_param)
#                     boundary_outputs.append(boundary_output)
#
#                     # 更新索引
#                     current_idx += boundary_size
#
#         # 组合所有边界的输出
#         boundary_sum = torch.zeros_like(boundary_outputs[0])
#         for i, output in enumerate(boundary_outputs):
#             boundary_sum += output * self.boundary_weights[i].view(1, 1)
#
#         # 计算最终输出
#         output = torch.sum(trunk_output * boundary_sum, 1).reshape(-1, 1) + self.b_0
#
#         return {"model_in": coords_org, "model_out": output}#dd
#
