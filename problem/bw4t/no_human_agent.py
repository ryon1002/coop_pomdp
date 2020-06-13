import numpy as np

class Agent:
    def __init__(self, env, r_policies, task_graph):
        self.env = env
        self.r_policies = r_policies
        self.task_graph = task_graph

    def set_policy(self, policy):
        self.policy = policy

    def set_task(self, t_id):
        self.task = self.task_graph.task_map[t_id]
        self.make_pi()

    def reset_belief(self):
        self.belief = np.ones(len(self.policy[0][0][0]))
        self.belief /= np.sum(self.belief)

    def update_belief(self, s, a_h):
        _, s_h = self.env.get_each_s(s)
        self.belief *= self.i_pi_h[s_h, a_h]
        self.belief /= np.sum(self.belief)

    def make_pi(self):
        pi_h = np.array([self.r_policies[gi] for _, gi in self.task.action_h]) # g, s, a
        s_hs = sorted(self.env.world.grids.keys())
        self.i_pi_h = np.zeros((len(s_hs), self.env.a_h, len(pi_h)))
        for s_h in s_hs:
            pi = pi_h[:, s_h, :].T # g|a
            s_pi = np.sum(pi, axis=1, keepdims=True)
            s_pi[s_pi == 0] = 1
            pi /= s_pi
            self.i_pi_h[s_h] = pi

    def act(self, s):
        v = [self._calc_q(v, self.belief) for a, v in sorted(self.policy[s].items())]
        return np.argmax(v)

    def _calc_q(self, a_vector, b):
        return np.max(np.dot(a_vector, b))

