import itertools
import numpy as np
from model.double_coop_irl_from import CoopIRL

class Correct(CoopIRL):
    def __init__(self, p_kind, p_num, objects, preferences):
        s = (p_num + 1) ** p_kind
        self.shape = [p_num + 1] * p_kind
        self.objects = [np.array(o) for o in objects]
        self.preferences = np.array([p for p in preferences])
        super().__init__(s, p_kind, p_kind, len(preferences), len(objects))

    def _check_complete(self, item, obj):
        lack = item - obj
        # print(lack, item, obj)
        lack_ids = lack < 0
        lack_sums = np.sum(lack[lack_ids]) * -1
        if lack_sums == 0:
            return ()
        if lack_sums > 2:
            return (0, 0, 0)
        if lack_sums == 1:
            return (np.argmax(lack_ids),)
        if np.sum(lack_ids) == 1:
            l_id = np.argmax(lack_ids)
            return (l_id, l_id)
        return tuple(np.where(lack_ids)[0])

    def _set_tro(self):
        shape_list = [range(i) for i in self.shape]
        action_list = [range(a) for a in [self.a_r, self.a_h]]
        for i in itertools.product(*shape_list):
            s = np.ravel_multi_index(i, self.shape, "clip")
            for th_h, objs in enumerate(self.objects):
                items = np.array(i)
                lack_items = set(self._check_complete(items, o) for o in objs)
                # print(lack_items)
                lack_items = sorted(lack_items, key=lambda x: len(x))
                for a_r, a_h in itertools.product(*action_list):
                    items = np.array(i)
                    items[a_r] += 1
                    items[a_h] += 1
                    ns = np.ravel_multi_index(items, self.shape, "clip")
                    self.t[a_r, a_h, s, ns] = 1
                    for l_item in lack_items:
                        if len(l_item) == 0:
                            break
                        if len(l_item) == 1 and l_item[0] == a_r:
                            self.r[a_r, a_h, s, :, th_h] = self.preferences[:, a_r] + 20
                            break
                        if len(l_item) == 2 and (l_item == (a_h, a_r) or l_item == (a_r, a_h)):
                            self.r[a_r, a_h, s, :, th_h] = \
                                np.sum(self.preferences[:, [a_h, a_r]], axis=1) + 20
                            break
                        self.r[a_r, a_h, s, :, th_h] = \
                            np.sum(self.preferences[:, [a_h, a_r]], axis=1)
                        break

