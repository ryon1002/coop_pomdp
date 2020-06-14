import numpy as np
from algo.policy_util import make_poilcy

a_dir = {0: np.array([-1, 0]), 1: np.array([0, -1]), 2: np.array([0, 1]), 3: np.array([1, 0])}


class BW4TCoopMDPSet(object):
    def __init__(self, pomdp, world, beta, task_graph):
        self.pomdp = pomdp
        self.world = world
        penalty = task_graph.penalty
        policies = {k: make_poilcy(v, beta) for k, v in self.world.single_q.items()}
        self.re_policies = self.reshape_policies(policies)

        # self.re_policies[-1] = {s: {k: s for k in range(100)}
        #                         for s in range(len(self.pomdp.i_s_map))}

        # self.v_map, self.v_for_t_map = self.make_v_map(task_graph, penalty)
        self.v_map, self.v_for_t_map = self.make_v_map(task_graph, penalty)
        self.v_map_full, self.v_for_t_map_full = self.make_v_map(task_graph)
        self.b_map = self.make_belief_map(task_graph, self.v_map_full, self.v_for_t_map_full)
        # self.b_map_for_agent = self.make_belief_map(task_graph, self.v_map, self.v_for_t_map,
        #                                             penalty)
        self.b_map_for_agent = self.make_belief_map(task_graph, self.v_map, self.v_for_t_map,
                                                    penalty)

    def make_v_map(self, task_graph, penalty=()):
        v_map = {s: np.zeros(self.pomdp.s) for s in range(len(task_graph.task_map))}
        v_for_t_map = {}
        for t_id in reversed(range(len(task_graph.task_network))):
            task = task_graph.task_map[t_id]

            v_for_t = self.calc_v(task, v_map, penalty)
            v = np.max(np.array([ov for ov in v_for_t.values()]), axis=0)

            v_map[t_id] = v
            v_for_t_map[t_id] = v_for_t
        return v_map, v_for_t_map

    def calc_v(self, task, v_map, penalty=()):
        valid_h = [va[1] for va in task.action_h if va[1] not in [10, -1]]
        valid_r = [va[1] for va in task.action_r if va[1] not in [10, -1]]

        v = {}
        for c, i in task.action:
            if i == -1:
                continue
            p = 20 if c == "h" and i != 10 and task.blocks[i] in penalty else 0
            n_t = task.next[(c, i)]
            if n_t:
                if c == "h":
                    ns_h = self.world.goals_id[i]
                elif c == "r":
                    ns_r = self.world.goals_id[i]
            one_v = np.zeros(len(self.pomdp.s_map))
            dist = self.world.dist[i]
            for s, (s_r, s_h) in self.pomdp.s_map.items():
                d = dist[s_h] if c == "h" else dist[s_r]
                if not n_t:
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
                    r = max(v_map[n_t.id][ns])
                one_v[s] = r - d
            v[(c, i)] = one_v - p
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

    def make_belief_map(self, task_graph, v_map_woa, v_map, penalty=()):
        b_map = {}
        # for t_id in [0]:
        for t_id in range(len(task_graph.task_network)):
            task = task_graph.task_map[t_id]
            b = np.zeros((len(task.action_r), self.pomdp.s, len(task.action_h)))
            for a_r, (_, g_r) in enumerate(task.action_r):
                for a_h, (_, g_h) in enumerate(task.action_h):
                    if g_r != 10 and g_r == g_h:
                        b[a_r, :, a_h] = -1000
                        continue
                    if g_r != 10 and g_r == g_h:
                        b[a_r, :, a_h] = -1000
                        continue
                    for s, (s_r, s_h) in self.pomdp.s_map.items():
                        p = 0
                        if g_r == -1:
                            b[a_r, s, a_h] = self.world.dist[g_h][s_h]
                            continue
                        if g_h == -1:
                            b[a_r, s, a_h] = self.world.dist[g_r][s_r]
                            continue

                        n_task = task
                        d_g_r = self.world.dist[g_r][s_r]
                        d_g_h = self.world.dist[g_h][s_h]
                        v, step = 100, 0
                        t_g_r, t_g_h, t_s_r, t_s_h, t_d_g_r, t_d_g_h = \
                            g_r, g_h, s_r, s_h, d_g_r, d_g_h
                        if t_d_g_r <= d_g_h:
                            while t_d_g_r + step <= d_g_h:
                                t_s_r, t_s_h, t_s = self._t_step(t_s_r, t_s_h, t_g_r, g_h, t_d_g_r)
                                step += t_d_g_r
                                n_task = n_task.next[("r", t_g_r)]
                                if n_task is None:
                                    break
                                t_g_r = self._next_goal(n_task.id, t_s, n_task.action_r, g_h, v_map)
                                if t_g_r is None:
                                    break
                                t_d_g_r = self.world.dist[t_g_r][t_s_r]

                            if step == d_g_h:
                                if n_task is not None:
                                    n_task = n_task.next[("h", g_h)]
                                    if n_task is not None:
                                        if g_h != 10 and task.blocks[g_h] in penalty:
                                            p += 20
                                        v = v_map_woa[n_task.id][t_s]
                            elif n_task is not None:
                                v = v_map[n_task.id][("h", g_h)][t_s]

                        else:
                            while t_d_g_h + step < d_g_r:
                                t_s_r, t_s_h, t_s = self._t_step(t_s_r, t_s_h, g_r, t_g_h, t_d_g_h)
                                step += t_d_g_h
                                n_task = n_task.next[("h", t_g_h)]
                                if t_g_h != 10 and task.blocks[t_g_h] in penalty:
                                    p += 20
                                if n_task is None:
                                    break
                                t_g_h = self._next_goal(n_task.id, t_s, n_task.action_h, g_r, v_map)
                                if t_g_h is None:
                                    break
                                t_d_g_h = self.world.dist[t_g_h][t_s_h]
                            if n_task is not None:
                                v = v_map[n_task.id][("r", g_r)][t_s]
                        b[a_r, s, a_h] = v - step - p
            b_map[t_id] = b
        return b_map

    def _t_step(self, s_r, s_h, g_r, g_h, dist):
        s_r = self.re_policies[g_r][s_r][dist]
        s_h = self.re_policies[g_h][s_h][dist]
        return s_r, s_h, self.pomdp.i_s_map[(s_r, s_h)]

    def _next_goal(self, t_id, s, c_g, o_g, v_map):
        if c_g[0][1] == -1:
            return None
        q = {a[1]: v_map[t_id][a][s] for a in c_g if a[1] == 10 or a[1] != o_g}
        if len(q) == 0:
            return None
        return max(q, key=q.get)

    def make_best_goals_for_js(self):
        return {t_id: self._calc_best_goal(b_map) for t_id, b_map in
                self.b_map_for_agent.items()}

    def _calc_best_goal(self, v_map):
        # return np.argmax(np.max(v_map, axis=0), axis=1).tolist()
        return np.argmax(np.max(v_map, axis=2), axis=0).tolist()
