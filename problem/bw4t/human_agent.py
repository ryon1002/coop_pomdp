import numpy as np
from scipy.special import softmax

class Human:
    def __init__(self, env, env_set, r_policies, task_graph):
        self.env = env
        self.env_set = env_set
        self.task_graph = task_graph
        self.r_policies = r_policies
        self.target = None

    def set_task(self, t_id):
        self.target_map = self.env_set.b_map[t_id]
        self.task = self.task_graph.task_map[t_id]
        self.make_pi()

    def set_r_policies(self, r_policies):
        self.r_policies = r_policies

    def reset_belief(self):
        self.belief = np.ones(len(self.policy[0][0][0]))
        self.belief /= np.sum(self.belief)

    def update_belief(self, s, a_r):
        if self.target is None:
            s_r, _ = self.env.get_each_s(s)
            prob_g = self.i_pi_r[s_r, a_r]
            target = self.target_map[:, s, :]
            target = softmax(target, axis=1)
            v = np.sum(target * prob_g[:, np.newaxis], axis=0)
            target_id = np.argmax(v)
            self.target = self.task.action_h[target_id][1]

    def make_pi(self):
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
