import itertools
import numpy as np
import heapq
from .single_mdp import BW4TSingleMDP

# a_map = {0: "^", 1: "<", 2: ">", 3: "v"}
a_dir = {0: np.array([-1, 0]), 1: np.array([0, -1]), 2: np.array([0, 1]), 3: np.array([1, 0])}
i_a_dir = {(-1, 0): 0, (0, -1): 1, (0, 1): 2, (1, 0): 3}


class BW4T:
    def __init__(self):
        self.shape = (14, 11)
        self._make_world_data()
        self.s = len(self.grids)

    def _make_world_data(self):
        self.map = np.ones(self.shape, dtype=np.int) * 15
        # self.map[0, :] &= 14
        # self.map[:, 0] &= 13
        # self.map[:, -1] &= 11

        self.map[:4, :] &= 0
        self.map[4, :] &= 14
        self.map[4:, 0] &= 13
        self.map[4:, -1] &= 11

        # self.map[:8, :] &= 0
        # self.map[8, :] &= 14
        # self.map[8:, 0] &= 13
        # self.map[8:, -1] &= 11

        # self.map[:12, :] &= 0
        # self.map[12:, 2] &= 13
        # self.map[12:, -3] &= 11
        # self.map[12, :] &= 14
        # self.map[12, [0, 1, -2, -1]] = 0

        self.map[-2, :] &= 7
        self.map[-1, :] &= 0

        # # Additional Wall
        # self.map[8, 3] &= 11
        # self.map[8, 4] &= 13

        self.goals = {}
        for g, (y, x) in enumerate(itertools.product([1, 5, 9], [1, 4, 7])):
        # for g, (y, x) in enumerate(itertools.product([5, 9], [1, 4, 7])):
        # for g, (y, x) in enumerate(itertools.product([9], [1, 4, 7])):
            self.map[y:y + 3, x:x + 3] = 0
            self.map[y:y + 3, x - 1] &= 11
            self.map[y:y + 3, x + 3] &= 13
            self.map[y - 1, x:x + 3] &= 7
            self.map[y + 3, x:x + 3] &= 14
            self.map[y + 2, x + 1] |= 8
            self.map[y + 3, x + 1] |= 1
            self.goals[g] = (y + 2, x + 1)
        self.map[12, 5] |= 8
        self.map[13, 5] |= 1
        self.goals[10] = (13, 5)

        self.grids = {i: np.array(v) for i, v in enumerate(zip(*np.where(self.map != 0)))}
        self.i_grids = {tuple(v): k for k, v in self.grids.items()}
        self.i_goals = {self.i_grids[v]:k for k, v in self.goals.items()}
        self.goals_id= {v:k for k, v in self.i_goals.items()}
        self.valid_moves = {k: self._make_valid_moves(self.map[v[0], v[1]]) for k, v in
                            self.grids.items()}
        self.transition = {}
        for s, valid_a in self.valid_moves.items():
            n_a = {}
            for a in valid_a:
                n_p = self.grids[s] + a_dir[a]
                n_a[a] = self.i_grids[tuple(n_p)]
            self.transition[s] = n_a

        self.dist = {g:self.calc_min_dist(self.i_grids[p]) for g, p in self.goals.items()}
        self.dist = {g:self.calc_min_dist(self.i_grids[p]) for g, p in self.goals.items()}

        self.single_q = BW4TSingleMDP(self).get_all_q_values()

    def _make_valid_moves(self, valid):
        return np.where(np.array([valid & 1 > 0,
                                  valid & 2 > 0,
                                  valid & 4 > 0,
                                  valid & 8 > 0]))[0]

    def calc_min_dist(self, start_s):
        q = [(0, start_s)]
        dist = {start_s:0}
        while len(q) > 0:
            v, s = heapq.heappop(q)
            for n_s in self.transition[s].values():
                if n_s not in dist:
                    heapq.heappush(q, (v + 1, n_s))
                    dist[n_s] = v + 1
        return dist

    def is_goal(self, s):
        return self.i_goals.get(s, -1)

    def print_world(self):
        print(self.map)

    # def show_world(self):
    # for yi, xi in itertools.product(range(self.map.shape[0]), range(self.map.shape[1])):
    #     color = "w" if self.map[yi, xi] > 0 else "brown"
    #     plt.gca().add_patch(patches.Rectangle((xi, -yi), 1, -1, facecolor=color))
    # for e in self.state.r_enemys:
    #     plt.gca().add_patch(patches.Rectangle(e[::-1] * np.array([1, -1]), 1, -1,
    #                                           facecolor="pink"))
    # for e in self.state.b_enemys:
    #     plt.gca().add_patch(patches.Rectangle(e[::-1] * np.array([1, -1]), 1, -1,
    #                                           facecolor="lightblue"))
    # plt.gca().add_patch(patches.Rectangle(self.state.human[::-1] * np.array([1, -1]), 1, -1,
    #                                       facecolor="r"))
    # plt.gca().add_patch(patches.Rectangle(self.state.agent[::-1] * np.array([1, -1]), 1, -1,
    #                                       facecolor="g"))
    # plt.ylim((-self.map.shape[0], 0))
    # plt.xlim((0, self.map.shape[1]))
    # plt.show()
