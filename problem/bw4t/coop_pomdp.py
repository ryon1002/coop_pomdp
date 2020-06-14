from model.coop_pomdp_from import CoopPOMDP
import numpy as np
import itertools
from scipy.special import softmax
from algo.policy_util import make_poilcy
from algo import util
from scipy.sparse import csr_matrix
from collections import defaultdict

a_dir = {0: np.array([-1, 0]), 1: np.array([0, -1]), 2: np.array([0, 1]), 3: np.array([1, 0])}


class BW4TCoopPOMDP(CoopPOMDP):
    def __init__(self, world, h_beta, r_beta):
        self.world = world
        self.greedy_policies = {k: self._make_greedy_map(v) for k, v in self.world.single_q.items()}
        self.h_policies = {k: make_poilcy(v, h_beta) for k, v in self.world.single_q.items()}
        self.h_policies[-1] = np.ones_like(self.h_policies[0])
        self.h_policies[-1] /= np.sum(self.h_policies[-1], axis=1, keepdims=True)
        self.r_policies = {k: make_poilcy(v, r_beta) for k, v in self.world.single_q.items()}
        self.r_policies[-1] = np.ones_like(self.r_policies[0])
        self.r_policies[-1] /= np.sum(self.r_policies[0], axis=1, keepdims=True)
        self.prepared = False

        self.s_map = {i: s for i, s in enumerate(itertools.product(world.grids.keys(), repeat=2))}
        self.i_s_map = {v: k for k, v in self.s_map.items()}
        self.i_s_map_r = {s: [] for s in world.grids.keys()}
        self.i_s_map_h = {s: [] for s in world.grids.keys()}
        for s, (s_r, s_h) in self.s_map.items():
            self.i_s_map_r[s_r].append(s)
            self.i_s_map_h[s_h].append(s)
        print(len(self.s_map))
        super().__init__(len(self.s_map) + 1, 4, 4, 1)
        self.base_t = self.convert_to_spearce(self.t, self.a_r, self.a_h)
        delattr(self, "t")
        # self.base_t = np.copy(self.t)
        # self.base_r = np.copy(self.r)

    def _make_greedy_map(self, q):
        policy = make_poilcy(q, np.inf)
        return {s: int(np.argmax(policy[s])) for s in range(len(policy))}

    def _set_tro(self):
        self.t[:, :, -1, -1] = 1
        # self.r[:, :, -1, :] = 0
        self.r_base = np.zeros_like(self.r)
        for s, (s_r, s_h) in self.s_map.items():
            c_a_r, c_a_h = self.world.valid_moves[s_r], self.world.valid_moves[s_h]
            for a_r, a_h in itertools.product(range(4), repeat=2):
                if a_r in c_a_r and a_h in c_a_h:
                    n_s_r, n_s_h = \
                        self.world.transition[s_r][a_r], self.world.transition[s_h][a_h]
                    self.r_base[a_r, a_h, s, :] = -1
                    n_s = self.i_s_map[(n_s_r, n_s_h)]
                    self.t[a_r, a_h, s, n_s] = 1
                else:
                    self.t[a_r, a_h, s, -1] = 1
                    self.r_base[a_r, a_h, s, :] = -1000
        # for s, (s_r, s_h) in self.s_map.items():

    def _pre_calc(self):
        if not self.prepared:
            return
        self.sum_r = np.zeros((self.a_r, self.s, self.th))
        for th in range(self.th):
            for s in range(self.s):
                self.sum_r[:, s, th] = np.sum(self.o[th, :, s] * self.r[:, :, s, th], axis=1)

        self.ns = {
            s: {a_r: {a_h: self._ex_all_nx(s, a_r, a_h) for a_h in range(self.a_h)} for a_r in
                range(self.a_r)} for s in range(self.s)}
        # P(a_h|s, a_r; theta)
        # self.update = np.zeros((self.a_r, self.a_h, self.s, self.th))
        self.update = np.zeros((self.a_r, self.a_h, self.s, self.th, self.th))
        for s in range(self.s):
            for a_r in range(self.a_r):
                # for th in range(self.th):
                # self.update[a_r, :, s, :] = np.dot(self.o[:, a_r, s].T, self.i_pi_r[s, a_r])
                #     print()
                #     self.update[a_r, :, s, th] = self.o[th, a_r, s]
                for th in range(self.th):
                    # self.update[a_r, :, s, th, th] = self.o[th, a_r, s]
                    self.update[a_r, :, s, th, :] = np.outer(self.o[th, a_r, s], self.i_pi_r[s, a_r, :, th])
        # print(self.update[a_r, :, 851, 0, :])
        # print(self.update[a_r, :, 851, 1, :])
        # print(self.update[a_r, :, 851, 2, :])
        # print(self.update[a_r, :, 851, 3, :])
        # exit()

        # self.update = np.zeros((self.a_r, self.a_h, self.s, self.th))
        # for s in range(self.s):
        #     for a_r in range(self.a_r):
        #         # self.update[a_r, :, s, :] = self.o[:, a_r, s].T
        #         if np.sum(self.i_pi_r[s, a_r]) != 0:
        #             # self.update[a_r, :, s, :] = np.dot(self.o[:, a_r, s]).T
        #             self.update[a_r, :, s, :] = np.dot(self.o[:, a_r, s].T, self.i_pi_r[s, a_r])
        #         else:
        #             self.update[a_r, :, s, :] = self.o[:, a_r, s].T
        #         # for th in range(self.th):
        #         #     self.update[a_r, :, s, th] = self.o[th, a_r, s]
        #         # for th2 in range(self.th):
        #             # self.update[a_r, :, s, th] += self.o[th, a_r, s] * self.i_pi_r[th]
        self.t = self.convert_to_spearce(self.t, self.a_r, self.a_h)
        # self.update = self.convert_to_spearce(self.update, self.a_r, self.a_h)

    def convert_to_spearce(self, arr, dim0, dim1):
        ret = {}
        for d0 in range(dim0):
            ret[d0] = {}
            for d1 in range(dim1):
                ret[d0][d1] = csr_matrix(arr[d0, d1])
        return ret

    def convert_to_dense(self, arr, dim0, dim1):
        ret = np.zeros((dim0, dim1, self.s, self.s))
        for d0 in range(dim0):
            for d1 in range(dim1):
                ret[d0, d1] = arr[d0][d1].todense()
        return ret

    def calc_a_vector(self, d=1, bs=None, with_a=True, algo=0):
        if d == 1:
            # self.a_vector = {s: self.sum_r[:, s, :].copy() for s in range(self.s)}
            self.a_vector = {s: self.sum_r[:, s, :].copy() for s in range(self.s)}
            return
        self.calc_a_vector(d - 1, bs, False, algo)
        a_vector = {}
        for s in range(self.s):
            a_vector[s] = {}
            for a_r in range(self.a_r):
                p_a_vector = []
                p_a_vector_nums = []
                for a_h in range(self.a_h):
                    ns = self.ns[s][a_r][a_h][0][0]

                    # if algo == 0:
                    #     tmp_p_a_vector = self.a_vector[ns] * self.update[a_r, a_h, s] + \
                    #                      self.o[:, a_r, s, a_h] * self.r[a_r, a_h, s]
                    # elif algo == 1:
                    #     # tmp = np.dot(self.a_vector[ns], self.i_pi_r[s, a_r]) * self.update[a_r, a_h, s]
                    #     # tmp = np.dot(np.dot(self.a_vector[ns], self.i_pi_r[s, a_r]), self.update[a_r, a_h, s])
                    #     # tmp = np.dot(self.a_vector[ns], self.update[a_r, a_h, s])
                    #     tmp = np.dot(self.a_vector[ns], self.update[a_r, a_h, s].T)
                    #     tmp_p_a_vector = tmp + self.o[:, a_r, s, a_h] * self.r[a_r, a_h, s]
                    #     # tmp_p_a_vector = tmp + self.r[a_r, a_h, s]
                    tmp = np.dot(self.a_vector[ns], self.update[a_r, a_h, s].T)
                    tmp_p_a_vector = tmp + self.o[:, a_r, s, a_h] * self.r[a_r, a_h, s]

                    p_a_vector.append(util.unique_for_raw(tmp_p_a_vector))
                    # p_a_vector.append(np.unique(tmp_p_a_vector, axis=0))
                    p_a_vector_nums.append(len(p_a_vector[-1]))
                a_vector_a = np.zeros((np.prod(p_a_vector_nums), self.th))
                # for m, i in enumerate(itertools.product(*[range(l) for l in p_a_vector_nums])):
                #     a_vector_a[m] = np.sum([p_a_vector[n][j] for n, j in enumerate(i)], axis=0)
                for m, i in enumerate(itertools.product(*p_a_vector)):
                    a_vector_a[m] = np.sum(i, axis=0)
                a_vector[s][a_r] = util.unique_for_raw(a_vector_a)
        print(d)
        if with_a:
            self.a_vector_a = {s: {a_r: util.prune(vector, bs) for a_r, vector in vectorA.items()}
                               for s, vectorA in
                               a_vector.items()} if bs is not None else a_vector
        else:
            self.a_vector = {s: util.prune(np.concatenate(list(vector.values()), axis=0), bs) for
                             s, vector in
                             a_vector.items()} if bs is not None else a_vector

    def set_world(self, env_set, task_graph, t_id, penalty):
        self.t = self.convert_to_dense(self.base_t, self.a_r, self.a_h)

        task = task_graph.task_map[t_id]
        task_net = task_graph.task_network[t_id]
        target = task.action
        h_target = [ht[1] for ht in task.action_h]
        r_target = [ht[1] for ht in task.action_r]

        self.th = len(h_target)
        self.r = np.empty((self.a_h, self.a_r, self.s, self.th))
        for i in range(self.th):
            self.r[:, :, :, i] = self.r_base[:, :, :, 0]
        for c, g in target:
            if g == -1:
                continue
            g_s = self.world.goals_id[g]
            n_t_id = task_net[(c, g)]
            if n_t_id == -1:
                next_v = np.ones((self.s)) * 100
            else:
                next_v = env_set.v_map_full[task_net[(c, g)]]
            if c == "r":
                for s_h in range(self.world.s):
                    s = self.get_s_from_id(g_s, s_h)
                    idx = np.where(self.t[:, :, :, s])
                    tmp = np.copy(self.t[idx[0], idx[1], idx[2], s])
                    self.t[idx[0], idx[1], idx[2], s] = 0
                    self.t[idx[0], idx[1], idx[2], -1] = tmp
                    g_h = self.world.i_goals.get(s_h, None)
                    if g_h in h_target and (g_h == 10 or g_h != g) and n_t_id != -1:
                        nn_task = task.next[(c, g)].next[("h", g_h)]
                        r = 100 if nn_task is None else env_set.v_map_full[nn_task.id][s]
                        if g_h != 10 and task.blocks[g_h] in penalty:
                            r -= 20
                        self.r[idx[0], idx[1], idx[2], :] = r
                    else:
                        self.r[idx[0], idx[1], idx[2], :] = next_v[s]
            elif c == "h":
                for s_r in range(self.world.s):
                    if self.world.i_goals.get(s_r, None) in r_target:
                        continue
                    s = self.get_s_from_id(s_r, g_s)
                    idx = np.where(self.t[:, :, :, s])
                    tmp = np.copy(self.t[idx[0], idx[1], idx[2], s])
                    self.t[idx[0], idx[1], idx[2], s] = 0
                    self.t[idx[0], idx[1], idx[2], -1] = tmp
                    r = next_v[s]
                    if g != 10 and task.blocks[g] in penalty:
                        r -= 20
                    self.r[idx[0], idx[1], idx[2], :] = r

        self.o = np.zeros((self.th, self.a_r, self.s, self.a_h))  # pi
        for gi, g in enumerate(h_target):
            for s, (_s_r, s_h) in self.s_map.items():
                self.o[gi, :, s] = self.h_policies[g][s_h]

        b_map = softmax(env_set.b_map[t_id], axis=2)
        # self.r_policies = {k: make_poilcy(v, np.inf) for k, v in self.world.single_q.items()}
        # self.r_policies[-1] = np.ones_like(self.r_policies[0])
        # self.r_policies[-1] /= np.sum(self.r_policies[-1], axis=1, keepdims=True)
        if len(r_target) > 0:
            pi_r = np.array([self.r_policies[gi] for gi in r_target])  # g, s, a
        else:
            pi_r = self.r_policies[-1][np.newaxis]
        self.i_pi_r = np.zeros((self.s, self.a_r, self.th, self.th))
        for s, (s_r, _s_h) in self.s_map.items():
            # pi = np.copy(pi_r[:, s_r, :].T)  # g|a
            # s_pi = np.sum(pi, axis=1, keepdims=True)
            # s_pi[s_pi == 0] = 1
            # pi /= s_pi
            for a_r in range(self.a_r):
                p = np.copy(pi_r[:, s_r, :].T[a_r])# g|a
                d = np.sum(p) if np.sum(p) != 0 else 1
                p /= d
                t = np.dot(p, b_map[:, s, :])
                self.i_pi_r[s, a_r] = np.tile(t, (self.th, 1)).T

        # l = 1
        l = 0.8
        # l = 0.5
        if l < 1:
            for s in range(self.i_pi_r.shape[0]):
                for a_r in range(self.i_pi_r.shape[1]):
                    tmp = l * self.i_pi_r[s, a_r] + (1 - l) * np.identity(len(task.action_h))
                    self.i_pi_r[s, a_r] = tmp / np.sum(tmp, axis=0, keepdims=True)

        o = np.copy(self.o)
        self.o = np.zeros((self.th, self.a_r, self.s, self.a_h))  # pi
        for s in range(self.s):
            for a_r in range(self.a_r):
                for th in range(self.th):
                    self.o[th, a_r, s] = np.dot(self.i_pi_r[s, a_r, :, th], o[:, a_r, s, :])

        self.prepared = True
        self._pre_calc()

    def get_s(self, pos_r, pos_h):
        return self.i_s_map[(self.world.i_grids[pos_r], self.world.i_grids[pos_h])]

    def get_s_from_id(self, id_s_r, id_s_h):
        return self.i_s_map[(id_s_r, id_s_h)]

    def get_pos(self, s):
        s_r, s_h = self.get_each_s(s)
        return tuple(self.world.grids[s_r]), tuple(self.world.grids[s_h])

    def get_each_s(self, s):
        return self.s_map[s]

    def get_next_s(self, s, a_r, a_h):
        if a_h is None:
            s_r, s_h = self.get_each_s(s)
            n_s_r = self.world.transition[s_r][a_r]
            return self.get_s_from_id(n_s_r, s_h)
        if a_r is None:
            s_r, s_h = self.get_each_s(s)
            n_s_h = self.world.transition[s_h][a_h]
            return self.get_s_from_id(s_r, n_s_h)
        return np.argmax(self.base_t[a_r][a_h][s])

    def check_goal(self, s):
        s_r, s_h = self.get_each_s(s)
        g_r, g_h = self.world.is_goal(s_r), self.world.is_goal(s_h)
        ret = []
        if g_r != -1:
            ret.append(("r", g_r))
        if g_h != -1:
            ret.append(("h", g_h))
        return ret

    def _make_2d_expand_dict(self, input):
        ret = defaultdict(dict)
        for (i, j), k in input.items():
            ret[str(i)][str(j)] = k
        return dict(ret)

    def make_base_info_for_js(self):
        self.world.map.tolist()
        return {
            "map": self.world.map.tolist(),
            "grid": self._make_2d_expand_dict(self.world.i_grids),
            "s_map": self._make_2d_expand_dict(self.i_s_map),
            "goal_s": self.world.i_goals,
        }

    def make_best_goals_for_js(self):
        print()
