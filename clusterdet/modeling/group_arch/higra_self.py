import copy
import numpy as np
import bisect


class UnionFind(object):
    def __init__(self, num_vertices):
        self.fa = [i for i in range(num_vertices)]
        self.rank = np.zeros(num_vertices)

    def find(self, x):
        if x == self.fa[x]:
            return x
        else:
            self.fa[x] = self.find(self.fa[x])
            return self.fa[x]

    def link(self, i, j):
        if self.rank[i] > self.rank[j]:
            i, j = j, i
        elif self.rank[i] == self.rank[j]:
            self.rank[j] += 1
        self.fa[i] = j
        return j


class HorizontalCutExplorer(object):
    def __init__(self, tree, altitudes):
        self.m_num_regions_cuts = []
        self.m_altitudes_cuts = []
        self.m_range_nodes_cuts = []

        self.m_original_tree = copy.deepcopy(tree)
        if not self.is_sorted(altitudes):
            self.m_use_node_map = True
            self.m_sorted_tree, self.m_node_map = sort_hierarchy_with_altitudes(tree, altitudes)
            self.m_altitudes = altitudes[self.m_node_map]
            self.private_init(self.m_sorted_tree, self.m_altitudes)
        else:
            self.m_use_node_map = False
            self.m_altitudes = copy.deepcopy(altitudes)
            self.private_init(self.m_original_tree, self.m_altitudes)

    def private_init(self, tree, a):
        min_alt_children = tree.accumulate_parallel(a)
        # single region partition... edge case
        root_idx = tree.num_vertices() - 1
        self.m_num_regions_cuts.append(1)
        self.m_altitudes_cuts.append(float(a[root_idx]))
        self.m_range_nodes_cuts.append((-1, -1))
        range_start = root_idx
        range_end = root_idx
        num_regions = len(tree.children[root_idx])

        current_threshold = a[range_start]

        while current_threshold != 0 and range_start >= tree.num_leaves:
            while min_alt_children[range_end] >= current_threshold:
                range_end -= 1
            while a[range_start - 1] >= current_threshold:
                range_start -= 1
                num_regions += len(tree.children[range_start]) - 1

            current_threshold = a[range_start - 1]
            self.m_num_regions_cuts.append(num_regions)
            self.m_altitudes_cuts.append(float(current_threshold))
            self.m_range_nodes_cuts.append((range_start, range_end))

    def horizontal_cut_from_index(self, cut_index):
        num_regions = self.m_num_regions_cuts[cut_index]
        nodes = - np.ones(num_regions, dtype=int)
        if self.m_use_node_map:
            ct = self.m_sorted_tree
        else:
            ct = self.m_original_tree

        if cut_index == 0:
            nodes[0] = ct.num_vertices() - 1
        else:
            altitude = self.m_altitudes_cuts[cut_index]
            _range = self.m_range_nodes_cuts[cut_index]
            j = 0
            for i in range(_range[0], _range[1] + 1):
                for c in ct.children[i]:
                    if self.m_altitudes[c] <= altitude:
                        nodes[j] = c
                        j += 1
        if self.m_use_node_map:
            nodes = self.m_node_map[nodes]

        return HorizontalCutNodes(nodes, self.m_altitudes_cuts[cut_index])

    def horizontal_cut_from_num_regions(self, num_regions, at_least=True):
        # https://stackoverflow.com/a/37873955/7701908
        pos = bisect.bisect_left(self.m_num_regions_cuts, num_regions)
        if pos == len(self.m_num_regions_cuts):
            cut_index = len(self.m_num_regions_cuts) - 1
        else:
            # TODO: double check
            cut_index = pos

        if self.m_num_regions_cuts[cut_index] > num_regions and (not at_least):
            if cut_index > 0:
                cut_index -= 1
        return self.horizontal_cut_from_index(cut_index)

    def is_sorted(self, a):
        for i in range(1, len(a)):
            if a[i - 1] > a[i]:
                return False
        return True


class HorizontalCutNodes(object):
    def __init__(self, nodes, altitude):
        self.nodes = nodes
        self.altitude = altitude

    def labelisation_leaves(self, tree):
        deleted = np.ones(tree.num_vertices(), bool)
        deleted[self.nodes] = False

        return self.reconstruct_leaf_data(tree, np.arange(tree.num_vertices()), deleted)

    def reconstruct_leaf_data(self, tree, altitudes, deleted_nodes=None):
        if deleted_nodes is None:
            raise NotImplementedError()
        reconstruction = self.propagate_sequential(tree, altitudes, deleted_nodes)
        leaf_weights = reconstruction[0:tree.num_leaves, ...]

        return leaf_weights

    def propagate_sequential(self, tree, altitudes, deleted_nodes):
        output = - np.ones_like(altitudes, dtype=float)
        for i in list(range(tree.num_vertices()))[::-1]:
            if i != tree.num_vertices() - 1 and deleted_nodes[i]:  # root node
                output[i] = output[tree.parents[i]]
            else:
                output[i] = altitudes[i]

        return output


class Tree(object):
    def __init__(self, parents, width):
        self.num_leaves = min(parents)  # `parents` is not always sorted. Using `min()` rather than `parents[0]`
        self.parents = parents
        self.children = [[] for _ in range(len(parents))]
        for i, p in enumerate(parents[
                              :-1]):  # skip the last special case, the root's parents is root itself, but its children don't contain itself.  # noqa
            self.children[int(p)].append(i)
        self.width = width

    def num_vertices(self):
        return len(self.parents)

    def num_edges(self):
        return len(self.parents) - 1

    def accumulate_sequential(self):
        # only support accumulate sum
        res = np.zeros(self.num_vertices())
        original = np.ones(self.num_vertices())
        boxes = np.zeros((self.num_vertices(), 4))
        for i, child in enumerate(self.children):
            if len(child) == 0:
                res[i] = 1
                boxes[i] = np.array([i % self.width, i//self.width,
                                     i % self.width, i//self.width])
            else:
                res[i] = sum(original[child])
                original[i] = res[i]
                boxes[i, 0] = boxes[child, 0].min()
                boxes[i, 1] = boxes[child, 1].min()
                boxes[i, 2] = boxes[child, 2].max()
                boxes[i, 3] = boxes[child, 3].max()

        return res, boxes

    def accumulate_parallel(self, values):
        # only support accumulate min
        assert len(values) == self.num_vertices()
        res = np.empty(self.num_vertices())
        for i, child in enumerate(self.children):
            if len(child) == 0:
                res[i] = 0
            else:
                res[i] = min(values[child])

        return res


class Graph(object):
    def __init__(self, graph, shape):
        assert isinstance(graph, np.ndarray)
        self.sources = graph[:, 0].astype(int)
        self.targets = graph[:, 1].astype(int)
        self.edge_weights = graph[:, 2]
        self.num_vertices = np.prod(shape)
        self.height, self.width = shape


class SubGraph(object):
    def __init__(self, graph, edge_indices):
        self.num_vertices = graph.num_vertices
        self.num_edges = len(edge_indices)
        self.sources = [graph.sources[i] for i in edge_indices]
        self.targets = [graph.targets[i] for i in edge_indices]
        self.edge_weights = [graph.edge_weights[i] for i in edge_indices]
        self.height, self.width = graph.height, graph.width


def create_edge(gradient, width, x1, y1, x2, y2):
    vertex_id = lambda x, y: y * width + x
    w = (gradient[y1, x1] + gradient[y2, x2]) / 2
    return vertex_id(x1, y1), vertex_id(x2, y2), w


def build_graph_naive(gradient, width, height):
    graph = []
    for y in range(height):
        for x in range(width):
            if x < width - 1:
                graph.append(create_edge(gradient, width, x, y, x + 1, y))
            if y < height - 1:
                graph.append(create_edge(gradient, width, x, y, x, y + 1))
    return np.array(graph)


def build_graphv2(gradient, width, height):
    vertex_id = lambda x, y: y * width + x
    raw = np.arange(width * height).reshape(height, width)

    g1 = np.concatenate(
        (raw[..., None], np.pad(raw[:, 1:], ((0, 0), (0, 1)), 'constant', constant_values=-2)[..., None]), axis=-1)
    g2 = np.concatenate(
        (raw[..., None], np.pad(raw[1:, :], ((0, 1), (0, 0)), 'constant', constant_values=-2)[..., None]), axis=-1)
    g3 = np.concatenate((g1, g2), axis=-1).reshape(-1, 2)
    g4 = g3[g3.min(axis=-1) > -1, :]

    new_gradient = np.array(gradient).flatten()
    diff = (new_gradient[g4[:, 0]] + new_gradient[g4[:, 1]]) / 2
    graph = np.concatenate((g4, diff[:, None]), axis=-1)
    return graph


def build_graph(gradient, width, height):
    raw = np.arange(width * height).reshape(height, width)

    g1 = np.concatenate((raw[:, :-1, None], raw[:, 1:, None]), axis=-1).reshape(-1, 2)
    g2 = np.concatenate((raw[:-1, :, None], raw[1:, :, None]), axis=-1).reshape(-1, 2)
    g3 = np.concatenate((g1, g2), axis=0)
    new_gradient = np.array(gradient).flatten()
    diff = (new_gradient[g3[:, 0]] + new_gradient[g3[:, 1]]) / 2
    graph = np.concatenate((g3, diff[:, None]), axis=-1)
    return graph


def bpt_canonical_from_sorted_edges(sources, targets, sorted_edge_indices, num_vertices):
    num_edge_mst = num_vertices - 1

    mst_edge_map = np.empty(num_edge_mst, dtype=np.int64)

    uf = UnionFind(num_vertices)

    roots = np.arange(num_vertices)
    parents = np.arange(num_vertices * 2 - 1)

    num_nodes = num_vertices
    num_edge_found = 0
    i = 0

    while num_edge_found < num_edge_mst and i < len(sorted_edge_indices):
        ei = sorted_edge_indices[i]
        c1 = uf.find(sources[ei])
        c2 = uf.find(targets[ei])
        if c1 != c2:
            parents[roots[c1]] = num_nodes
            parents[roots[c2]] = num_nodes
            newRoot = uf.link(c1, c2)
            roots[newRoot] = num_nodes
            mst_edge_map[num_edge_found] = ei
            num_nodes += 1
            num_edge_found += 1
        i += 1

    assert num_edge_found == num_edge_mst, "Input graph must be connected."

    return parents, mst_edge_map


def bpt_canonical(graph, edge_weights=None):
    if edge_weights is None:
        edge_weights = graph.edge_weights
        sorted_edges_indices = np.argsort(edge_weights)
    else:
        sorted_edges_indices = np.argsort(edge_weights)

    sources = graph.sources
    targets = graph.targets
    num_vertices = graph.num_vertices
    parents, mst_edge_map = bpt_canonical_from_sorted_edges(sources, targets, sorted_edges_indices, num_vertices)

    num_points = num_vertices

    levels = np.zeros_like(parents, dtype=float)
    levels[np.arange(num_points, len(levels))] = np.array(edge_weights)[mst_edge_map]

    return Tree(parents, graph.width), levels, mst_edge_map


def correct_attribute_BPT(tree, altitude, attribute):
    result = np.empty_like(attribute, dtype=float)
    result[:tree.num_leaves] = 0
    non_leaf_start = tree.num_leaves
    root_idx = tree.num_vertices() - 1
    for n in range(non_leaf_start, root_idx):  # exclude leaves and root
        if altitude[n] != altitude[tree.parents[n]]:  # TODO:
            result[n] = attribute[n]
        else:
            maxc = - np.inf
            # children_iterator
            for c in tree.children[n]:
                maxc = max(maxc, 0 if c < non_leaf_start else result[c])
            result[n] = maxc
    result[root_idx] = attribute[root_idx]

    return result


def watershed_hierarchy_by_area(graph, canonize_tree=True):
    bpt, altitudes, mst_edge_map = bpt_canonical(graph)
    mst = SubGraph(graph, mst_edge_map)

    attribute_areas, boxes = bpt.accumulate_sequential()

    corrected_attribute = correct_attribute_BPT(bpt, altitudes, attribute_areas)
    persistence = bpt.accumulate_parallel(corrected_attribute)
    mst_edge_weights = persistence[bpt.num_leaves: bpt.num_vertices()]
    new_tree, new_altitudes, _ = bpt_canonical(mst, mst_edge_weights)

    if canonize_tree:
        new_tree, new_altitudes = canonize_hierarchy(new_tree, new_altitudes)

    attribute_areas, boxes = new_tree.accumulate_sequential()
    return attribute_areas, boxes
    return new_tree, new_altitudes


def simplify_tree(tree, criterion):
    n_nodes = tree.num_vertices()
    n_leaves = tree.num_leaves
    new_ranks = - np.ones(n_nodes, dtype=np.int64)
    new_ranks[:n_leaves] = np.arange(n_leaves)

    count = n_leaves
    for i in range(n_leaves, n_nodes - 1):
        if not criterion[i]:
            new_ranks[i] = count
            count += 1

    new_ranks[n_nodes - 1] = count
    count += 1

    new_parent = -np.ones(count, dtype=np.int64)
    node_map = -np.ones(count, dtype=np.int64)

    count = new_parent.size - 1
    new_parent[count] = count
    node_map[count] = n_nodes - 1
    count -= 1

    for i in list(range(n_nodes - 1))[::-1]:
        if (not criterion[i]) or (i < n_leaves):
            new_parent[count] = new_ranks[tree.parents[i]]
            node_map[count] = i
            count -= 1
        else:
            new_ranks[i] = new_ranks[tree.parents[i]]

    return Tree(new_parent, tree.width), node_map


def canonize_hierarchy(tree, altitudes, return_node_map=False):
    tree, node_map = simplify_tree(tree, altitudes == altitudes[tree.parents])
    new_altitudes = altitudes[node_map]
    if return_node_map:
        raise NotImplementedError()
    else:
        return tree, new_altitudes


def sort_hierarchy_with_altitudes(tree, altitudes):
    sorted = np.argsort(altitudes)
    reverse_sorted = np.empty_like(sorted, dtype=float)
    reverse_sorted[sorted] = np.arange(len(sorted))

    par = tree.parents
    new_par = np.empty_like(sorted, dtype=float)
    for i in range(len(reverse_sorted)):
        new_par[i] = reverse_sorted[par[sorted[i]]]

    return Tree(new_par), sorted


###############################################################################
# Test
###############################################################################
class TestCutHelper(object):
    def __init__(self):
        self.tree = Tree((11, 11, 11, 12, 12, 16, 13, 13, 13, 14, 14, 17, 16, 15, 15, 18, 17, 18, 18))
        altitudes = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 3, 1, 2, 3))
        self.py_ch = HorizontalCutExplorer(self.tree, altitudes)
        self.cut_nodes = (
            (18,),
            (17, 13, 14),
            (11, 16, 13, 14),
            (0, 1, 2, 3, 4, 5, 13, 9, 10)
        )

    def test_horizontal_cut_nodes(self):
        c = self.py_ch.horizontal_cut_from_num_regions(3)
        lbls = c.labelisation_leaves(self.tree)
        ref_lbls = (17, 17, 17, 17, 17, 17, 13, 13, 13, 14, 14)
        if np.all(lbls == ref_lbls):
            print("[Successes] Test passed!")
        else:
            print("[Fail]")
            print(lbls)


class TestWatershed(object):
    def __init__(self):
        shape = (1, 19)
        gradient = np.random.random(shape)
        _graph = build_graph(gradient, 19, 1)
        self.graph = Graph(_graph, shape=shape)
        self.graph.edge_weights = np.asarray((0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0))

    def test_watershed_hierarchy_by_area(self):
        ref_parents = np.asarray((19, 19, 20, 20, 20, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 23, 23, 23, 24,
                                  24, 25, 26, 26, 25, 27, 27, 27), dtype=np.int64)
        ref_altitudes = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 3, 5))

        tree, altitudes = watershed_hierarchy_by_area(self.graph)

        if np.allclose(ref_parents, tree.parents) and np.allclose(ref_altitudes, altitudes):
            print("[Successes] Test passed!")
        else:
            print("[Fail]")
            print(tree.parents)
            print(altitudes)


class TestDet(object):
    def __init__(self):
        import random
        random.seed(1)
        np.random.seed(1)
        import higra as hg
        shape = (400, 600)
        height, width = shape
        gradient = np.random.random(shape)

        hg_graph = hg.get_4_adjacency_graph(gradient.shape[:2])
        edge_weights = hg.weight_graph(hg_graph, gradient, hg.WeightFunction.mean)
        tree, altitudes = hg.watershed_hierarchy_by_area(hg_graph, edge_weights)
        cut_helper = hg.HorizontalCutExplorer(tree, altitudes)
        a1 = cut_helper.horizontal_cut_from_num_regions(1)
        import time

        start = time.time()
        _graph = build_graph(gradient, width=width, height=height)
        print("naive {}".format(time.time() - start))
        start = time.time()
        _graph2 = build_graphv2(gradient, width, height)
        print("v2 {}".format(time.time() - start))
        start = time.time()
        _graph22 = build_graph(gradient, width, height)
        print("v22 {}".format(time.time() - start))
        start = time.time()
        print("ChatGPT {}".format(time.time() - start))

        a2 = _graph2.tolist()
        a3 = _graph22.tolist()
        a2.sort(key=lambda x: x[-1])
        a3.sort(key=lambda x: x[-1])

        # assert np.allclose(_graph_gou, _graph)
        assert np.allclose(a2, a3)
        self.graph = Graph(_graph, shape=shape)
        py_tree, py_altitudes = watershed_hierarchy_by_area(self.graph)
        py_cuter = HorizontalCutExplorer(py_tree, py_altitudes)
        a2 = py_cuter.horizontal_cut_from_num_regions(1)
        print("flllll")


if __name__ == "__main__":
    # tester = TestCutHelper()
    # tester.test_horizontal_cut_nodes()
    # tester = TestWatershed()
    # tester.test_watershed_hierarchy_by_area()
    tester = TestDet()