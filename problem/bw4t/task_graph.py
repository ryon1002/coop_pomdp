import yaml
import copy
import json
import numpy as np

class Task:
    def __init__(self, blocks, goals, c):
        self.blocks = blocks
        self.c = c
        self.full_goals = goals
        self._make_actions()

    def _make_actions(self):
        self.action_h = tuple(self._get_valid_one_action("h"))
        if len(self.action_h) == 0:
            self.action_h = (("h", -1),)
        self.action_r = tuple(self._get_valid_one_action("r"))
        if len(self.action_r) == 0:
            self.action_r = (("r", -1),)
        self.action = self.action_h + self.action_r

    def id_string(self):
        return json.dumps({"blocks": self.blocks, "c": self.c, "goals": self.goals})

    def _get_valid_one_action(self, c):
        if self.c[c][0] != -1:
            return [(c, 10)]
        return [(c, i) for i, b in enumerate(self.blocks) if b != -1]

    def get_next_task(self, c, i):
        to = copy.deepcopy(self)
        if i == 10:
            if to.goals[0] == to.c[c][0]:
                to.goals = to.goals[1:]
                if len(to.goals) == 0:
                    return None
            else:
                to.blocks[to.c[c][1]] = to.c[c][0]
            to.c[c] = (-1, -1)
        else:
            to.c[c] = (to.blocks[i], i)
            to.blocks[i] = -1
        to._make_actions()
        return to

    def get_next_id(self, c, i):
        n_task = self.next[(c, i)]
        if n_task is None:
            return -1
        return n_task.id

    def get_next_task_for_js(self):
        return {"h" : {a[1]:self.get_next_id(*a) for a in self.action_h},
                "r" : {a[1]:self.get_next_id(*a) for a in self.action_r}}

    # def get_task_def_for_js(self):
    #     return {"h" : {a[1]:self.get_next_id(*a) for a in self.action_h}

class Task_NO(Task):
    def __init__(self, blocks, goals, c):
        self.blocks = blocks
        self.c = c
        self.goals = sorted(goals)
        self._make_actions()

    def get_next_task(self, c, i):
        to = copy.deepcopy(self)
        if i == 10:
            if to.c[c][0] in to.goals:
                to.goals.remove(to.c[c][0])
                if len(to.goals) == 0:
                    return None
            to.c[c] = (-1, -1)
        else:
            to.c[c] = (to.blocks[i], i)
            to.blocks[i] = -1
        to._make_actions()
        return to


class Task_NO_2(Task_NO):
    def _get_valid_one_action(self, c):
        if self.c[c][0] != -1:
            return [(c, 10)]
        if c == "r":
            return [(c, i) for i, b in enumerate(self.blocks) if b != -1]
        else:
            return [(c, i) for i, b in enumerate(self.blocks) if b != -1 and b not in self.forbid]


class TaskGraph:
    def __init__(self, filename, type=0):
        # data = yaml.load(open(filename, "r"), Loader=yaml.FullLoader)
        data = json.load(open(filename, "r"))
        # root = Task(data["blocks"], data["goals"], {"h": (-1, -1), "r": (-1, -1)})
        # if type == 1:
        root = Task_NO(data["blocks"], data["goals"], {"h": (-1, -1), "r": (-1, -1)})
        # else:
        #     root = Task_NO_2(data["blocks"], data["goals"], data["forbid"],
        #                      {"h": (-1, -1), "r": (-1, -1)})
        self.task_map = {0: root}
        task_id_map = {root.id_string(): 0}
        self.task_network = {}
        self.penalty = data["penalty"]

        current_id = 0
        while current_id < len(self.task_map):
            current_task = self.task_map[current_id]
            tmp_net = {}
            for c, i in current_task.action:
                n = current_task.get_next_task(c, i)
                if n is None:
                    tmp_net[(c, i)] = -1
                    continue
                n_str = n.id_string()
                if n_str not in task_id_map:
                    self.task_map[len(task_id_map)] = n
                    task_id_map[n_str] = len(task_id_map)
                next_id = task_id_map[n_str]
                tmp_net[(c, i)] = next_id
            self.task_network[current_id] = tmp_net
            current_id += 1

        for i, t in self.task_map.items():
            t.id = i
            t.next = {a: self.task_map[n] if n != -1 else None
                      for a, n in self.task_network[i].items()}

        self.i_task_depend = {i: [] for i in range(len(self.task_network))}
        for f, ns in self.task_network.items():
            for t in ns.values():
                if t != -1:
                    self.i_task_depend[t].append(f)

    def make_task_net_for_js(self):
        return {t_id:task.get_next_task_for_js() for t_id, task in self.task_map.items()}


