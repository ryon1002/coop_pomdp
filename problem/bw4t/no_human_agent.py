import joblib
import numpy as np
from scipy.special import softmax
from algo.policy_util import make_poilcy

class Agent:
    def __init__(self, env, env_set, policy_dir, h_beta):
        self.env = env
        self.env_set = env_set
        self.h_policies = {k:make_poilcy(v, h_beta) for k, v in self.env.world.single_q.items()}
        self.greedy_policies = env.greedy_policies
        self.policy_dir = policy_dir
        self.reset_belief_flag = False
        self.policies = {}

    def get_policy(self, t_id):
        if t_id not in self.policies:
            self.policies[t_id] = joblib.load(f"{self.policy_dir}/{t_id}.pkl")
        return self.policies[t_id]

    def set_task(self, task, s, first):
        self.prev_task = self.task if not first else None
        self.task = task
        self._make_pi()
        self.policy = self.get_policy(self.task.id)
        self.reset_belief(first)

    def _make_pi(self):
        s_hs = sorted(self.env.world.grids.keys())
        if self.task.action_h[0][1] == -1:
            self.i_pi_h = np.ones((len(s_hs), self.env.a_h, 1))
            return
        pi_h = np.array([self.h_policies[gi] for _, gi in self.task.action_h]) # g, s, a
        self.i_pi_h = np.zeros((len(s_hs), self.env.a_h, len(pi_h)))
        for s_h in s_hs:
            pi = pi_h[:, s_h, :].T # g|a
            s_pi = np.sum(pi, axis=1, keepdims=True)
            s_pi[s_pi == 0] = 1
            pi /= s_pi
            self.i_pi_h[s_h] = pi

    def act(self, s):
        if len(self.task.action_r) == 1:
            s_r = self.env.get_each_s(s)[0]
            return self.greedy_policies[self.task.action_r[0][1]][s_r]
        v = [self._calc_q(v, self.belief) for a, v in sorted(self.policy[s].items())]
        # print(v)
        # print(self.belief)
        return np.argmax(v)

    def _calc_q(self, a_vector, b):
        return np.max(np.dot(a_vector, b))

    def reset_belief(self, first):
        if first:
            self.belief = np.ones(len(self.policy[0][0][0]))
            self.belief /= np.sum(self.belief)
        else:
            if self.reset_belief_flag:
                self.belief = np.ones(len(self.policy[0][0][0]))
                self.belief /= np.sum(self.belief)
                self.reset_belief_flag = False
            else:
                n_id = [self.prev_task.action_h.index(t) for t in self.task.action_h]
                belief = self.belief[n_id]
                self.belief = belief / np.sum(belief)

    def set_belief(self, belief):
        self.belief = belief


    def update_belief(self, s, a_h):
        if a_h is None:
            v = np.array([self.env_set.v_for_t_map_full[self.task.id][a][s]
                          for a in self.task.action_h])
            b = softmax(v)
            self.belief = b
            return
        _, s_h = self.env.get_each_s(s)
        self.belief *= self.i_pi_h[s_h, a_h]
        self.belief /= np.sum(self.belief)
