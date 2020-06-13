from algo.vi import do_value_iteration
from model.coop_pomdp_from import CoopPOMDP
import numpy as np
import copy
import itertools
import json
from scipy.special import softmax

a_dir = {0: np.array([-1, 0]), 1: np.array([0, -1]), 2: np.array([0, 1]), 3: np.array([1, 0])}


class BW4TCoopPOMDP(CoopPOMDP):
    def __init__(self, world, h_policies, r_policies, d=0, target=-1):
        self.world = world
        self.h_policies = h_policies
        self.r_policies = r_policies
        self.prepared = False

        self.s_map = {i: s for i, s in enumerate(itertools.product(world.grids.keys(), repeat=2))}
        self.i_s_map = {v: k for k, v in self.s_map.items()}
        self.i_s_map_r = {s:[] for s in world.grids.keys()}
        self.i_s_map_h = {s:[] for s in world.grids.keys()}
        for s, (s_r, s_h) in self.s_map.items():
            self.i_s_map_r[s_r].append(s)
            self.i_s_map_h[s_h].append(s)
        print(len(self.s_map))
        super().__init__(len(self.s_map) + 1, 4, 4, 1)
        self.base_t = np.copy(self.t)
        # self.base_r = np.copy(self.r)

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
        # P(s', a_h|s, a_r; theta)
        self.update = np.zeros((self.a_r, self.a_h, self.s, self.s, self.th))
        for th in range(self.th):
            for s in range(self.s):
                for a_r in range(self.a_r):
                    t = np.sum(self.t[a_r, :, s] * self.o[th, a_r, s][:, np.newaxis], axis=0)
                    self.update[a_r, :, s, :, th] = np.outer(self.o[th, a_r, s], t)

        # for s in range(self.s):
        # # for s in [759]:
        # # for s in [63]:
        #     for a_r in range(self.a_r):
        #         n_th = self.i_pi_r[s, a_r]
        #         if np.sum(n_th) == 0:
        #             continue
        #         update_org = self.update[a_r, :, s, :, :]
        #         # print(np.sum(update_org))
        #         update_new = np.zeros_like(update_org)
        #         for th in range(self.th):
        #             for th2 in range(self.th):
        #                 update_new[:, :, th] += update_org[:, :, th2] * n_th[th2]
        #         self.update[a_r, :, s, :, :] = update_new
        #         # print(np.sum(update_new))

        self.ns = {
            s: {a_r: {a_h: self._ex_all_nx(s, a_r, a_h) for a_h in range(self.a_h)} for a_r in
                range(self.a_r)} for s in range(self.s)}

    def set_world(self, env_set, task_graph, t_id):
        # import tracemalloc
        # tracemalloc.start()
        self.t = np.copy(self.base_t)

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
            g_s = self.world.goals_id[g]
            n_t_id = task_net[(c, g)]
            if n_t_id == -1:
                next_v = np.ones((self.s)) * 100
            else:
                next_v = env_set.v_map[task_net[(c, g)]]
            if c == "r":
                for s in self.i_s_map_r[g_s]:
                    idx = np.where(self.t[:, :, :, s])
                    tmp = np.copy(self.t[idx[0], idx[1], idx[2], s])
                    self.t[idx[0], idx[1], idx[2], s] = 0
                    self.t[idx[0], idx[1], idx[2], -1] = tmp
                    self.r[idx[0], idx[1], idx[2], :] = next_v[s]
            elif c == "h":
                for s in self.i_s_map_h[g_s]:
                    idx = np.where(self.t[:, :, :, s])
                    tmp = np.copy(self.t[idx[0], idx[1], idx[2], s])
                    self.t[idx[0], idx[1], idx[2], s] = 0
                    self.t[idx[0], idx[1], idx[2], -1] = tmp
                    self.r[idx[0], idx[1], idx[2], :] = next_v[s]

        self.o = np.zeros((self.th, self.a_r, self.s, self.a_h)) # pi
        for gi, g in enumerate(h_target):
            for s, (_s_r, s_h) in self.s_map.items():
                self.o[gi, :, s] = self.h_policies[g][s_h]

        # b_map = env_set.b_map[t_id]
        b_map = softmax(env_set.b_map[t_id], axis=2)
        # b_map = np.argmax(env_set.b_map[t_id], axis=2)
        pi_r = np.array([self.r_policies[gi] for gi in r_target]) # g, s, a
        # self.i_pi_r = np.zeros((self.s, self.a_r, self.th, self.th))
        # self.i_pi_r_2 = np.zeros((self.s, self.a_r, self.th))
        self.i_pi_r = np.zeros((self.s, self.a_r, self.th, self.th))
        # pi_mat = np.tile(self.i_pi_r[s, a_r], (self.th, 1)).T
        for s, (s_r, _s_h) in self.s_map.items():
            pi = pi_r[:, s_r, :].T # g|a
            s_pi = np.sum(pi, axis=1, keepdims=True)
            s_pi[s_pi == 0] = 1
            pi /= s_pi
            for g in range(len(pi_r)):
                # self.i_pi_r[s, :, b_map[g, s]] += pi[:, g]
                for th in range(self.th):
                    self.i_pi_r[s, :, th] += np.tile(pi[:, g] * b_map[g, s, th], (self.th, 1)).T
                    # self.i_pi_r_2[s, :, th] += pi[:, g] * b_map[g, s, th]

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
        return np.argmax(self.t[a_r, a_h, s])

    def check_goal(self, s):
        s_r, s_h = self.get_each_s(s)
        g_r, g_h = self.world.is_goal(s_r), self.world.is_goal(s_h)
        ret = []
        if g_r != -1:
            ret.append(("r", g_r))
        if g_h != -1:
            ret.append(("h", g_h))
        return ret


