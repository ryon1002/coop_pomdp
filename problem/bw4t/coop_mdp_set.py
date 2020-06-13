from model.coop_pomdp_from import CoopPOMDP
import numpy as np
import copy
import itertools
import json

a_dir = {0: np.array([-1, 0]), 1: np.array([0, -1]), 2: np.array([0, 1]), 3: np.array([1, 0])}


class BW4TCoopMDPSet(object):
    def __init__(self, pomdp, world, policies, task_graph):
        self.pomdp = pomdp
        self.world = world
        self.re_policies = self.reshape_policies(policies)
        # self.t = {}
        # for g, pi in policies.items():
        #     t = np.zeros((pomdp.a_r, pomdp.s, pomdp.s))
        #     for s, (s_r, s_h) in pomdp.s_map.items():
        #         a_h = np.argmax(pi[s_h])
        #         t[:, s] = pomdp.t[:, a_h, s]
        #     t[:, -1, -1] = 1
        #     self.t[g] = t
        self.v_map = {s: np.zeros(self.pomdp.s) for s in range(len(task_graph.task_map))}
        self.v_for_t_map = {}
        need_v_update = np.ones(len(task_graph.task_map), dtype=np.bool)
        check_v_map = {}
        while np.sum(need_v_update) > 0:
            t_id = np.max(np.where(need_v_update)[0])
            task_net = task_graph.task_network[t_id]
            task = task_graph.task_map[t_id]

            v_for_t = self.calc_v(task, task_net, self.v_map)
            v = np.max(np.array([ov for ov in v_for_t.values()]), axis=0)

            if not np.array_equal(v, self.v_map[t_id]):
                need_v_update[task_graph.i_task_depend[t_id]] = True
                self.v_map[t_id] = v
                self.v_for_t_map[t_id] = v_for_t
                check_v_map[t_id] = v.reshape((self.world.s, self.world.s))
            need_v_update[t_id] = False
        # print(check_v_map[0])
        # np.save("check.npy", check_v_map[0])
        # print(np.array_equal(np.load("check.npy"), check_v_map[0]))
        self.make_belief_map(task_graph)
        # exit()
        # print(t_id)

    def calc_v(self, task, target, v_map):
        valid_h = [va[1] for va in task.action_h if va[1] != 10]
        valid_r = [va[1] for va in task.action_r if va[1] != 10]

        v = {}
        for (c, i), n_t in target.items():
            one_v = np.zeros(len(self.pomdp.s_map))
            dist = self.world.dist[i]
            if n_t != -1:
                if c == "h":
                    ns_h = self.world.goals_id[i]
                elif c == "r":
                    ns_r = self.world.goals_id[i]
            for s, (s_r, s_h) in self.pomdp.s_map.items():
                d = dist[s_h] if c == "h" else dist[s_r]
                if n_t == -1:
                    r = 100
                else:
                    if c == "h":
                        if len(valid_r) == 0:
                            limit = 1 if i != 10 or s_r == self.world.goals_id[10] else 2
                            # limit = 1
                            od = min(d, len(self.re_policies[10][s_r]) - limit)
                            ns = [self.pomdp.i_s_map[(self.re_policies[10][s_r][od], ns_h)]]
                        else:
                            ns_r = [self.re_policies[g][s_r].get(d, self.world.goals_id[g]) for g in
                                    valid_r]
                            ns = [self.pomdp.i_s_map[(ns_r_, ns_h)] for ns_r_ in ns_r]
                    elif c == "r":
                        if len(valid_h) == 0:
                            od = min(d, len(self.re_policies[10][s_h]) - 1)
                            ns = [self.pomdp.i_s_map[(ns_r, self.re_policies[10][s_h][od])]]
                        else:
                            ns_h = [self.re_policies[g][s_h].get(d, self.world.goals_id[g]) for g in
                                    valid_h]
                            ns = [self.pomdp.i_s_map[(ns_r, ns_h_)] for ns_h_ in ns_h]
                    r = max(v_map[n_t][ns])
                one_v[s] = r - d
            v[(c, i)] = one_v
        return v

    def reshape_policies(self, policies):
        re_policies = {}
        for p_id, policy in policies.items():
            re_policy = {}
            g = self.world.i_grids[self.world.goals[p_id]]
            for s in range(len(policy) - 1):
                tmp_s = s
                count = 0
                re_policy_s = {0: tmp_s}
                while tmp_s != g:
                    a = np.argmax(policy[tmp_s])
                    tmp_s = self.world.transition[tmp_s][a]
                    count += 1
                    re_policy_s[count] = tmp_s
                    pass
                re_policy[s] = re_policy_s
            re_policies[p_id] = re_policy
        return re_policies

    # def make_belief_map(self, task_graph):
    #     self.b_map = {}
    #     for t_id in [0]:
    #         task = task_graph.task_map[t_id]
    #         b = np.zeros((len(task.action_r), self.pomdp.s, len(task.action_h)))
    #         for a_r, (c_r, g_r) in enumerate(task.action_r):
    #             for a_h, (c_h, g_h) in enumerate(task.action_h):
    #                 for s, (s_r, s_h) in self.pomdp.s_map.items():
    #                     # s = self.pomdp.get_s((10, 10), (10, 0))
    #                     # self.pomdp.s_map[s]
    #                     c_t_id = t_id
    #                     d_g_r = self.world.dist[g_r][s_r]
    #                     d_g_h = self.world.dist[g_h][s_h]
    #                     t_d_g_r, t_d_g_h, t_g_h, t_s_r, t_s_h = d_g_r, d_g_h, g_h, s_r, s_h
    #                     step = 0
    #                     while t_d_g_r > t_d_g_h:
    #                         step += t_d_g_h
    #                         t_s_r = self.re_policies[g_r][t_s_r][t_d_g_h]
    #                         t_s_h = self.world.goals_id[t_g_h]
    #                         t_s = self.pomdp.i_s_map[(t_s_r, t_s_h)]
    #                         c_t_id = task_graph.task_network[c_t_id][(c_h, t_g_h)]
    #
    #                         if (c_r, g_r) not in task_graph.task_network[c_t_id]:
    #                             break
    #
    #                         q = {a:self.v_for_t_map[c_t_id][a][t_s] for a in task_graph.task_map[c_t_id].action_h}
    #                         _, t_g_h = max(q, key=q.get)
    #
    #                         t_d_g_r = self.world.dist[g_r][t_s_r]
    #                         t_d_g_h = self.world.dist[t_g_h][t_s_h]
    #                     if (c_r, g_r) not in task_graph.task_network[c_t_id]:
    #                         b[a_r, s, a_h] = self.v_map[c_t_id][t_s] - step
    #                     else:
    #                         t_s_r = self.world.goals_id[g_r]
    #                         t_s_h = self.re_policies[t_g_h][t_s_h][t_d_g_r]
    #                         t_s = self.pomdp.i_s_map[(t_s_r, t_s_h)]
    #                         c_t_id = task_graph.task_network[c_t_id][(c_r, g_r)]
    #                         if (c_h, t_g_h) in task_graph.task_network[c_t_id]:
    #                             b[a_r, s, a_h] = self.v_for_t_map[c_t_id][(c_h, t_g_h)][t_s] - d_g_r
    #                         else:
    #                             b[a_r, s, a_h] = self.v_map[c_t_id][t_s] - d_g_r
    #                     # elif t_d_g_r == t_d_g_h:
    #                     #     t_s_r = self.world.goals_id[g_r]
    #                     #     t_s_h = self.world.goals_id[t_g_h]
    #                     #     t_s = self.pomdp.i_s_map[(t_s_r, t_s_h)]
    #                     #     if (c_r, g_r) in task_graph.task_network[c_t_id]:
    #                     #         c_t_id = task_graph.task_network[c_t_id][(c_r, g_r)]
    #                     #     if (c_h, t_g_h) in task_graph.task_network[c_t_id]:
    #                     #         c_t_id = task_graph.task_network[c_t_id][(c_h, t_g_h)]
    #                     #     b[a_r, s, a_h] = self.v_map[c_t_id][t_s] - d_g_r
    #                     # else:
    #                     #     t_s_r = self.world.goals_id[g_r]
    #                     #     t_s_h = self.re_policies[t_g_h][t_s_h][t_d_g_r]
    #                     #     t_s = self.pomdp.i_s_map[(t_s_r, t_s_h)]
    #                     #     # if (c_r, g_r) in task_graph.task_network[c_t_id]:
    #                     #     #     c_t_id = task_graph.task_network[c_t_id][(c_r, g_r)]
    #                     #     # c_t_id = task_graph.task_network[c_t_id][(c_r, g_r)]
    #                     #     b[a_r, s, a_h] = self.v_for_t_map[c_t_id][(c_r, g_r)][t_s] - s
    #         self.b_map[t_id] = b
    #         print(b[:, 613, :])

    def make_belief_map_bak(self, task_graph):
        self.b_map = {}
        for t_id in [0]:
            task = task_graph.task_map[t_id]
            b = np.zeros((len(task.action_r), self.pomdp.s, len(task.action_h)))
            for a_r, (c_r, g_r) in enumerate(task.action_r):
                for a_h, (c_h, g_h) in enumerate(task.action_h):
                    for s, (s_r, s_h) in self.pomdp.s_map.items():
                        # s = self.pomdp.get_s((10, 10), (10, 0))
                        # self.pomdp.s_map[s]
                        c_t_id = t_id
                        d_g_r = self.world.dist[g_r][s_r]
                        d_g_h = self.world.dist[g_h][s_h]
                        if d_g_r <= d_g_h:
                            t_s_r = self.world.goals_id[g_r]
                            t_s_h = self.re_policies[g_h][s_h][d_g_r]
                            t_s = self.pomdp.i_s_map[(t_s_r, t_s_h)]
                            c_t_id = task_graph.task_network[c_t_id][(c_r, g_r)]
                            b[a_r, s, a_h] = self.v_map[c_t_id][t_s] - d_g_r
                        else:
                            t_s_r = self.re_policies[g_r][s_r][d_g_h]
                            t_s_h = self.world.goals_id[g_h]
                            t_s = self.pomdp.i_s_map[(t_s_r, t_s_h)]
                            c_t_id = task_graph.task_network[c_t_id][(c_h, g_h)]
                            b[a_r, s, a_h] = self.v_map[c_t_id][t_s] - d_g_h
            self.b_map[t_id] = np.argmax(b, axis=2)
            # print(b[:, 759, :])

    def make_belief_map(self, task_graph):
        self.b_map = {}
        # for t_id in [0]:
        for t_id in range(len(task_graph.task_network)):
            task = task_graph.task_map[t_id]
            b = np.zeros((len(task.action_r), self.pomdp.s, len(task.action_h)))
            for a_r, (_, g_r) in enumerate(task.action_r):
                for a_h, (_, g_h) in enumerate(task.action_h):
                    if g_r != 10 and g_r == g_h:
                        b[a_r, :, a_h] = -1000
                        continue
                    for s, (s_r, s_h) in self.pomdp.s_map.items():
                        # s = self.pomdp.get_s((10, 10), (10, 0))
                        # self.pomdp.s_map[s]
                        t_t_id = t_id
                        d_g_r = self.world.dist[g_r][s_r]
                        d_g_h = self.world.dist[g_h][s_h]
                        step = 0
                        t_g_r, t_g_h, t_s_r, t_s_h, t_d_g_r, t_d_g_h = \
                            g_r, g_h, s_r, s_h, d_g_r, d_g_h
                        if t_d_g_r <= d_g_h:
                            while t_d_g_r + step < d_g_h:
                                t_s_r = self.world.goals_id[t_g_r]
                                t_s_h = self.re_policies[g_h][t_s_h][t_d_g_r]
                                t_s = self.pomdp.i_s_map[(t_s_r, t_s_h)]
                                t_t_id = task_graph.task_network[t_t_id][("r", t_g_r)]
                                step += t_d_g_r
                                if t_t_id == -1:
                                    break
                                c_t_g_r = task_graph.task_map[t_t_id].action_r
                                q = {a[1]: self.v_for_t_map[t_t_id][a][t_s] for a in c_t_g_r
                                     if a[1] == 10 or a[1] != g_h}
                                t_g_r = max(q, key=q.get)
                                t_d_g_r = self.world.dist[t_g_r][t_s_r]
                            if t_t_id == -1:
                                b[a_r, s, a_h] = 100 - step
                            else:
                                b[a_r, s, a_h] = self.v_for_t_map[t_t_id][("h", g_h)][t_s] - step
                        else:
                            while t_d_g_h + step < d_g_r:
                                t_s_r = self.re_policies[g_r][t_s_r][t_d_g_h]
                                t_s_h = self.world.goals_id[t_g_h]
                                t_s = self.pomdp.i_s_map[(t_s_r, t_s_h)]
                                t_t_id = task_graph.task_network[t_t_id][("h", t_g_h)]
                                step += t_d_g_h
                                if t_t_id == -1:
                                    break
                                c_t_g_h = task_graph.task_map[t_t_id].action_h
                                q = {a[1]: self.v_for_t_map[t_t_id][a][t_s] for a in c_t_g_h
                                     if a[1] == 10 or a[1] != g_r}
                                t_g_h = max(q, key=q.get)
                                t_d_g_h = self.world.dist[t_g_h][t_s_h]
                            if t_t_id == -1:
                                b[a_r, s, a_h] = 100 - step
                            else:
                                b[a_r, s, a_h] = self.v_for_t_map[t_t_id][("r", g_r)][t_s] - step
        # self.b_map[t_id] = np.argmax(b, axis=2)
        # self.b_map[t_id] = np.argmax(b, axis=2)
            self.b_map[t_id] = b
        # print(b[:, 759, :])

# def _t_step(self, t_id, s, ):
