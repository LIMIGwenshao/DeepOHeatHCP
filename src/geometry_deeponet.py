import numpy as np
from src.dataio_utils import (
    sample_grf_model,
    sample_training_data_single_domain,
    sample_sensor_as_coords_train_data_single_domain,
    sample_eval_data_single_domain,
    find_boundaries_endpoints,
    find_set_by_range,
)
from src.geometry_utils import iterate_over_entire_geometry

class Cuboid(object):

    def __init__(
        self,
        domain,
        starting_idx=0,
        parent=None,
        parent_boundary=None,
        dim=3,
        mesh=False,
    ):

        self.domain, self.domain_step = domain, domain.copy()
        self.name = self.domain["domain_name"]
        self.boundaries_list = ["front", "back", "left", "right", "bottom", "top"]

        if parent is not None and parent_boundary is None:#检查报错设计，用于确保程序稳定
            raise ValueError(
                f"When creating node with designated parent, one need to provide which boundary this child lies on."
            )

        if parent is not None:
            parent.add_child(parent_boundary, self)
        else:
            self.parent = parent #刚开始时为none
            self.parent_boundary = parent_boundary #刚开始为none

        self.children = {key: [] for key in self.boundaries_list}
        self.starting_idx = starting_idx

        self.sensors, self.tensor, self.power_map, self.conductivity = (
            None,
            None,
            None,
            None,
        )
        self.whole_set, self.inside_set, self.pde_set = set(), set(), set()
        self.boundaries_set = {key: set() for key in self.boundaries_list}

        _, srf = sample_grf_model(dim=dim)
        self.sample(srf, dim=dim, mesh=mesh)#这里不同
        self.ending_idx = self.starting_idx + self.tensor.shape[0]

    def sample(self, srf, dim=3, mesh=False): #cuboid类中用于获取数据的函数。用于训练阶段，生成完整的训练数据
        if not mesh:
            (
                self.sensors,
                self.tensor,
                self.power_map,
                self.inside_set,
                self.boundaries_set,
                self.conductivity,
            ) = sample_training_data_single_domain(
                domain=self.domain_step,
                srf=srf,
                dim=dim,
                starting_idx=self.starting_idx,
            )
        else: #网格模式（mesh=True）#生成数据集的核心
            self.sensors, self.tensor, self.boundaries_set, self.conductivity = (
                sample_sensor_as_coords_train_data_single_domain(#获取数据的核心步骤
                    domain=self.domain_step, srf=srf, dim=dim
                )
            )

    def sample_grid_points(self, dim=3): #用于评估阶段，只生成评估所需的基本数据
        self.sensors, self.tensor = sample_eval_data_single_domain(
            domain=self.domain_step, dim=dim
        ) #return sensors, eval_coords # sensor一维数组，每个元素对应一个点的温度值；tensor是坐标
        self.power_map, self.conductivity = np.array([]), np.array([])

    def find_inside_set(self, whole_set, boundaries_set):
        return whole_set - set().union(*boundaries_set.values())

    def update_set(self): #这个函数主要是调用各种赋值特殊点集的函数
        self.whole_set = set(np.arange(self.starting_idx, self.ending_idx))
        self.boundaries_set = self.find_boundries_set(
            self.domain_step, self.tensor, self.boundaries_set
        )#边界点集的操作最多
        self.inside_set = self.find_inside_set(self.whole_set, self.boundaries_set)
        self.pde_set = self.inside_set

    def find_boundries_set(self, domain, tensor, boundaries_set):

        def find_node_boundary_set(node, boundary_set, adjacent_boundary_name):#这个函数用来处理单个边界
            node_boundary_endpoints = find_boundaries_endpoints(
                node.domain_step["geometry"]["starts"],
                node.domain_step["geometry"]["ends"],
            )#返回一个字典，包含每个边界的起始以及终止水平
            starts, ends = (
                node_boundary_endpoints[adjacent_boundary_name]["starts"],
                node_boundary_endpoints[adjacent_boundary_name]["ends"],
            )#返回该边界的起始点以及终止点
            #print(boundary_set)
            boundary_points = tensor[list(boundary_set), :]# 结果是一个新的二维张量，只包含边界点的坐标
            return find_set_by_range(boundary_points, min(boundary_set), starts, ends)

        def find_single_boundary_set(boundary_name):
            if not domain[boundary_name]["bc"]:
                return set()#如果某个边界没有边界条件（boundary condition），则返回一个空集合。

            boundary_name_idx = self.boundaries_list.index(boundary_name)
            adjacent_boundary_name = (
                self.boundaries_list[boundary_name_idx - 1]# 如果是奇数，选择这个
                if boundary_name_idx % 2# 条件：索引是奇数
                else self.boundaries_list[boundary_name_idx + 1]# 如果是偶数，选择这个
            )

            boundary_set = boundaries_set[boundary_name]
            adj_boundary_set = set()
            if not self.is_root() and self.parent_boundary == adjacent_boundary_name:
                adj_boundary_set |= find_node_boundary_set(
                    self.parent,
                    boundaries_set[boundary_name],  # 修改这里
                    adjacent_boundary_name
                )

            if not self.is_leaf() and self.children[boundary_name] != []:
                for child_node in self.children[boundary_name]:
                    adj_boundary_set |= find_node_boundary_set(
                        child_node,
                        boundaries_set[boundary_name],  # 修改这里
                        adjacent_boundary_name
                    )

            # if not self.is_root() and self.parent_boundary == adjacent_boundary_name:
            #
            #     adj_boundary_set |= find_node_boundary_set(
            #         self.parent, boundary_name, adjacent_boundary_name
            #     )
            #
            # if not self.is_leaf() and self.children[boundary_name] != []:
            #
            #     for child_node in self.children[boundary_name]:
            #         adj_boundary_set |= find_node_boundary_set(
            #             child_node, boundary_name, adjacent_boundary_name
            #         )

            boundary_set -= adj_boundary_set

            return boundary_set

        return {key: find_single_boundary_set(key) for key in self.boundaries_list}#每个边界的索引

    def add_child(self, boundary_name, child_node):
        self.children[boundary_name].append(child_node)
        child_node.parent = self
        child_node.parent_boundary = boundary_name

    def to_children(self, boundary_name):
        return self.children[boundary_name]

    def to_child(self, boundary_name, idx):
        return self.children[boundary_name][idx]

    def to_parent(self):
        return self.parent

    def if_last_sibling(self):
        if not self.parent:
            return True, None, None

        self_idx = self.parent.children[self.parent_boundary].index(self)
        if self_idx != len(self.parent.children[self.parent_boundary]) - 1:
            return False, self.parent_boundary, self_idx + 1
        else:
            parent_boundary_idx = self.boundaries_list.index(self.parent_boundary)

            if parent_boundary_idx == len(self.boundaries_list) - 1:
                return True, None, None

            for parent_boundary in self.boundaries_list[parent_boundary_idx + 1 :]:
                if self.parent.children[parent_boundary] != 0:
                    return False, parent_boundary, 0

            return True, None, None

    def to_next_sibling(self, boundary_name, next_idx):
        return self.parent.children[boundary_name][next_idx]

    def to_root(self):
        if self.is_root():
            return self

        prev = self.to_parent()

        while prev.parent is not None:
            prev = prev.to_parent()

        return prev

    def is_root(self):
        return True if self.parent is None else False

    def is_leaf(self):
        for boundary_name in self.boundaries_list:
            if self.children[boundary_name] != []:
                return False

        return True


def create_stacking_cuboidal_geometry(domains_list, dim=3, mesh=False):
    # 初始化根节点为None
    root = None
    # 遍历所有域，寻找根节点
    for domain in domains_list:
        # 如果还没有找到根节点，且当前域被标记为根节点
        if root is None and domain["node"]["root"]:
            root = domain
        # 如果已经找到根节点，且又发现一个被标记为根节点的域，报错
        elif root is not None and domain["node"]["root"]:
            raise ValueError(f"Only one node can be assigned as the root node")

    # 如果没有找到根节点，报错
    if not root:
        raise ValueError(f"Missing the root node, fail to create correct eometry")

    # 使用根域创建根节点Cuboid对象
    root_node = Cuboid(root, dim=dim, mesh=mesh)

    # 如果根节点是叶节点，直接返回
    if root["node"]["leaf"]:
        return root_node

    # 初始化变量
    all_leaf = False  # 标记是否所有节点都是叶节点
    current_node = root_node  # 当前处理的节点
    child_starting_idx = current_node.ending_idx  # 子节点的起始索引

    # 当不是所有节点都是叶节点时，继续处理
    while not all_leaf:
        # 初始化变量
        children_all_leaf = True  # 标记当前节点的所有子节点是否都是叶节点
        stem_child_boundary_name, stem_child_idx = None, None  # 存储非叶节点的子节点信息

        # 遍历当前节点的所有边界及其子节点列表
        for boundary_name, children_idx_list in current_node.domain["node"]["children"].items():
            # 遍历每个边界上的子节点索引
            for child_idx in children_idx_list:
                # 创建新的子节点
                new_child_node = Cuboid(
                    domains_list[child_idx],
                    starting_idx=child_starting_idx,
                    dim=dim,
                    mesh=mesh,
                )
                # 将新节点添加为当前节点的子节点
                current_node.add_child(boundary_name, new_child_node)
                # 更新下一个子节点的起始索引
                child_starting_idx = new_child_node.ending_idx

                # 如果找到非叶节点，记录其信息
                if not domains_list[child_idx]["node"]["leaf"] and children_all_leaf:
                    stem_child_boundary_name, stem_child_idx = (
                        boundary_name,
                        len(current_node.children[boundary_name]) - 1,
                    )

                # 更新所有子节点是否都是叶节点的标记
                children_all_leaf &= domains_list[child_idx]["node"]["leaf"]

        # 如果当前节点有非叶子节点，继续处理该子节点
        if not children_all_leaf:
            current_node = current_node.to_child(
                stem_child_boundary_name, stem_child_idx
            )
        else:
            # 获取当前节点是否是最后一个兄弟节点，以及下一个兄弟节点的信息
            last_sibling, next_sibling_boundary, next_sibling_idx = (
                current_node.if_last_sibling()
            )

            # 如果当前节点是最后一个兄弟节点，且不是根节点，向上回溯
            while last_sibling and current_node.parent is not None:
                current_node = current_node.to_parent()
                last_sibling, next_sibling_boundary, next_sibling_idx = (
                    current_node.if_last_sibling()
                )

            # 如果回溯到根节点且是最后一个兄弟节点，结束处理
            if last_sibling:
                all_leaf = True
            else:
                # 否则，移动到下一个兄弟节点
                current_node = current_node.to_next_sibling(
                    next_sibling_boundary, next_sibling_idx
                )

    # 返回到根节点
    current_node = current_node.to_root()
    # 返回构建好的几何结构
    return current_node


def fetch_data(geometry, mode="train", srf=None, dim=3, var=1, len_scale=0.3):
    sensors_list, tensor_list, power_map_list, conductivity_list = [], [], [], []

    if srf is None:
        _, srf = sample_grf_model(dim=dim, var=var, len_scale=len_scale)

    def fetch_single_node(node):
        if mode == "train":
            node.sample(srf=srf, dim=dim)
        elif mode == "eval":
            node.sample_grid_points(dim=dim)

        sensors_list.append(node.sensors)
        tensor_list.append(node.tensor)
        power_map_list.append(node.power_map)
        conductivity_list.append(node.conductivity)

    iterate_over_entire_geometry(geometry, fetch_single_node)
    # print(sensors_list)

    return (
        np.concatenate(sensors_list, 0),
        #np.concatenate(sensors_list, 0)[:441],  # 只保留前441个值
        np.concatenate(tensor_list, 0),
        np.concatenate(power_map_list, 0),
        np.concatenate(conductivity_list, 0),
    )


def fetch_mesh_data(geometry, mode="train", srf=None, dim=3, var=1, len_scale=0.3): #在数据集获取数据的调用中，最终是调用到这个函数
    sensors_list, tensor_list, conductivity_list = [], [], []

    if srf is None:
        _, srf = sample_grf_model(dim=dim, var=var, len_scale=len_scale) #return seed, srf，seed是高斯采样的结果，srf是方法，这里只返回方法 srf = SRF(model, seed=seed)

    def fetch_single_node(node): #这个函数是在每个node都会使用的
        if mode == "train":
            node.sample(srf=srf, dim=dim, mesh=True) #node就是cubic类的实例化，只不过是根结点的
        elif mode == "eval":
            node.sample_grid_points(dim=dim)

        sensors_list.append(node.sensors)
        tensor_list.append(node.tensor)
        conductivity_list.append(node.conductivity)

    iterate_over_entire_geometry(geometry, fetch_single_node) #沿着几何逐个迭代 def iterate_over_entire_geometry(geometry, fun, *args):

    return (
        #np.concatenate(sensors_list, 0)[:441],  # 只保留前 441 个值
        np.concatenate(sensors_list, 0), #原始操作
        #np.concatenate(sensors_list, 0)[-36:],  # 只保留后 121 个值
        np.concatenate(tensor_list, 0),
        np.concatenate(conductivity_list, 0),
    ) #由这里返回数据用于训练


