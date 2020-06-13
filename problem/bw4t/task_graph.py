import yaml
import copy
import json


class Task:
    def __init__(self, blocks, goals, c):
        self.blocks = blocks
        self.c = c
        self.goals = goals
        self._make_actions()

    def _make_actions(self):
        self.action_h = tuple(self._get_valid_one_action("h"))
        self.action_r = tuple(self._get_valid_one_action("r"))
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


class TaskGraph:
    def __init__(self, filename):
        data = yaml.load(open(filename, "r"), Loader=yaml.FullLoader)
        root = Task(data["blocks"], data["goals"], {"h": (-1, -1), "r": (-1, -1)})
        self.task_map = {0: root}
        # task_id_map = defaultdict(lambda x : len(task_id_map))
        self.task_id_map = {root.id_string():0}
        # task_id_map[root.id_string()] = 0
        self.task_network = {}

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
                if n_str not in self.task_id_map:
                    self.task_map[len(self.task_id_map)] = n
                    self.task_id_map[n_str] = len(self.task_id_map)
                next_id = self.task_id_map[n_str]
                tmp_net[(c, i)] = next_id
            self.task_network[current_id] = tmp_net
            current_id += 1

        self.i_task_depend = {i:[] for i in range(len(self.task_network))}
        for f, ns in self.task_network.items():
            for t in ns.values():
                if t != -1:
                    self.i_task_depend[t].append(f)

