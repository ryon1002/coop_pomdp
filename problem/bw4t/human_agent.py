import numpy as np
from scipy.special import softmax
from algo.policy_util import make_poilcy

class Human:
    def __init__(self, env, env_set, r_beta):
        self.env = env
        self.env_set = env_set
        self.r_policies = {k:make_poilcy(v, r_beta) for k, v in self.env.world.single_q.items()}

    def set_task(self, task, s, first):
        self.prev_task = self.task if not first else None
        self.task = task
        self.target_map = self.env_set.b_map[self.task.id]
        self._make_pi()
        self.reset_target(first, s)

    def _make_pi(self):
        pi_r = np.array([self.r_policies[gi] for _, gi in self.task.action_r]) # g, s, a
        s_rs = sorted(self.env.world.grids.keys())
        self.i_pi_r = np.zeros((len(s_rs), self.env.a_r, len(pi_r)))
        for s_r in s_rs:
            pi = pi_r[:, s_r, :].T # g|a
            s_pi = np.sum(pi, axis=1, keepdims=True)
            s_pi[s_pi == 0] = 1
            pi /= s_pi
            self.i_pi_r[s_r] = pi

    def act(self, s):
        _, s_h = self.env.get_each_s(s)
        policy = self.r_policies[self.target][s_h]
        return np.argmax(policy)

    def _calc_q(self, a_vector, b):
        return np.max(np.dot(a_vector, b))

    def reset_target(self, first, s):
        if first:
            self.target = None
            self.r_target = None
        else:
            self.update_target_for_new_task(s)

    def update_target_for_new_task(self, s):
        if len(self.task.action_r) == 1:
            self.r_target = 0
        if len(self.task.action_h) == 1:
            self.target = self.task.action_h[0][1]
            return
        if self.target not in [a_h[1] for a_h in self.task.action_h]:
            self.target = None


    def _d_dist(self, s, g):
        return self.env.world.dist[g][s] + self.env.world.dist[10][self.env.world.goals_id[g]]

    def update_target(self, s, a_r):
        if len(self.task.action_h) == 1:
            self.target = self.task.action_h[0][1]
            return
        # if self.r_targets is None:
        #     s_r, _ = self.env.get_each_s(s)
        #     prob_g = self.i_pi_r[s_r, a_r]
        #     self.r_targets = list(np.where(prob_g == np.max(prob_g))[0])
        if self.target is None or self.need_update_r_target(s, a_r):
            self.update_r_target(s, a_r)
            target = self.target_map[self.r_target, s, :]
            v_prob = softmax(target * 0.1)
            target_id = np.random.choice(np.arange(len(v_prob)), p=v_prob)
            # target_id = np.argmax(v_prob)

            # v = np.sum(target * prob_g[:, np.newaxis], axis=0)
            # v_prob = v / np.sum(v)
            # target_id = np.random.choice(np.arange(len(v_prob)), p=v_prob)
            # target_id = np.argmax(v)

            self.target = self.task.action_h[target_id][1]

    def need_update_r_target(self, s, a_r):
        if self.r_target is None:
            return True
        s_r, _ = self.env.get_each_s(s)
        prob_g = self.i_pi_r[s_r, a_r]
        return np.prod(prob_g <= prob_g[self.r_target]) != 1

    def update_r_target(self, s, a_r):
        s_r, _ = self.env.get_each_s(s)
        prob_g = self.i_pi_r[s_r, a_r]
        r_v = np.array(
            [self.env_set.v_for_t_map_full[self.task.id][a][s] for a in self.task.action_r])
        r_v = softmax(r_v * 0.1)
        prob_g *= r_v
        prob_g /= np.sum(prob_g)
        self.r_target = np.random.choice(np.arange(len(prob_g)), p=prob_g)
        # self.r_target = np.argmax(prob_g)

