import joblib
import numpy as np
from collections import defaultdict

from problem.bw4t.human_agent import Human
from problem.bw4t.no_human_agent import Agent


def episode(env, agent, human, s, task, first_a_h, log=False):
    agent.set_task(task, s, True)
    human.set_task(task, s, True)
    if log:
        print()
        print("target", human.target)
    change_task = False
    add_count = 0
    goal_list = []
    if first_a_h is not None:
        agent.update_belief(s, first_a_h)
        s = env.get_next_s(s, None, first_a_h)
    #

    # agent.update_belief(s, None)
    for count in range(1000):
        goals = env.check_goal(s)
        for g in goals:
            if g in task.next:
                if g[0] == "h" and g[1] != 10 and task.blocks[g[1]] == 3:
                    add_count += 20
                if g[0] == "h":
                    agent.reset_belief_flag = True
                    # add_count += 20
                task = task.next[g]
                goal_list.append(g)
                if task is None:
                    if log:
                        print("---goal--- ", g, -1)
                    return count + add_count, tuple(goal_list)
                if log:
                    print("---goal--- ", g, task.id)
                change_task = True
        if change_task:
            agent.set_task(task, s, False)
            human.set_task(task, s, False)
            if log:
                print("target", human.target)
        a_r = agent.act(s)
        human.update_target(s, a_r)
        a_h = human.act(s)
        agent.update_belief(s, a_h)
        s = env.get_next_s(s, a_r, a_h)
        if log:
            print(a_r, a_h)
            print(env.get_pos(s))


if __name__ == '__main__':
    policy_dir, algo = "base", 0
    # policy_dir, algo = "own", 1

    result_dir = "results/bw4t/5"
    policy_dir = f"{result_dir}/policy_{policy_dir}/"

    base_env, task_graph, env_set = joblib.load(f"{result_dir}/env/base_env.pkl")
    task = task_graph.task_map[0]
    # start_s = base_env.get_s((4, 10), (12, 1))
    # start_s = base_env.get_s((4, 8), (6, 0))
    # start_s = base_env.get_s((7, 10), (8, 4))
    start_s, first_a_h = base_env.get_s((5, 10), (8, 2)), 2
    # start_s = base_env.get_s((4, 8), (6, 0))

    agent = Agent(base_env, env_set, policy_dir, 1)
    human = Human(base_env, env_set, 1)

    counts = []
    goals_dict = defaultdict(int)
    log = False
    # for _ in range(1):
    for _ in range(100):
    # for _ in range(1000):
        count, goals = episode(base_env, agent, human, start_s, task, first_a_h,  log)
        counts.append(count)
        goals_dict[(count, goals)] += 1
        # if count > 24:
        #     print("break", count)
        #     break
        # print()
    for k, v in sorted(goals_dict.items(), key=lambda x:x[1], reverse=True):
        print(v, k)
    print(np.mean(counts))
