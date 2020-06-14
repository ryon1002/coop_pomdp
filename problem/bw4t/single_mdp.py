import numpy as np
import itertools
from algo.vi import do_value_iteration

a_dir = {0: np.array([-1, 0]), 1: np.array([0, -1]), 2: np.array([0, 1]), 3: np.array([1, 0])}


class BW4TSingleMDP(object):
    def __init__(self, world):
        self.world = world
        self.s = len(self.world.grids) + 1
        self.t = np.zeros((11, 4, self.s, self.s))
        self.r = np.zeros((11, 4, self.s, self.s))

        self.t[:, :, -1, -1] = 1
        self.r[:, :, -1, :] = 0
        for s, a in itertools.product(range(self.s - 1), range(4)):
            if a in self.world.valid_moves[s]:
                # n_p = self.world.grids[s] + a_dir[a]
                n_s = self.world.transition[s][a]
                g = self.world.i_goals.get(n_s, None)
                self.t[:, a, s, n_s] = 1
                self.r[:, a, s, n_s] = -1
                if g is not None:
                    self.t[g, a, s, n_s] = 0
                    self.t[g, a, s, -1] = 1
                    self.r[g, a, s, n_s] = 0
                    self.r[g, a, s, -1] = 100
            else:
                self.t[:, a, s, -1] = 1
                self.r[:, a, s, -1] = -1000

    def get_all_q_values(self):
        return {i: do_value_iteration(self.t[i], self.r[i]).T for i in self.world.i_goals.values()}

