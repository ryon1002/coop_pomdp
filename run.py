import matplotlib.pyplot as plt
import numpy as np

from algo.double_coop_irl_3 import CoopIRL
from algo.vi import do_value_iteration
from problem.bw4t.world import BW4T
from problem.bw4t.task_graph import TaskGraph
from problem.bw4t.coop_pomdp import BW4TCoopPOMDP
from problem.bw4t.coop_mdp_set import BW4TCoopMDPSet
from problem.bw4t.single_mdp import BW4TSingleMDP
from problem.bw4t.no_human_agent import Agent
from problem.bw4t.human_agent import Human
import pickle
import datetime


def _make_belief(dim, n_max):
    if dim == 2:
        for i in range(n_max):
            yield [i, n_max - i]
    else:
        for i in range(n_max):
            for rest in _make_belief(dim - 1, n_max - i):
                yield [i] + rest


def make_belief(dim):
    if dim == 1:
        return np.array([[1]])
    b = [i for i in _make_belief(dim, 11)]
    b = np.array(b) * 0.1
    # b = [i for i in _make_belief(dim, 6)]
    # b = np.array(b) * 0.2
    return b


if __name__ == '__main__':
    algo = 1
    target = -1

    # world = BW4T()
    # task_graph = TaskGraph("problem/bw4t/map_data/map3.yaml")
    # # world.print_world()
    # # exit()
    # single_mdp = BW4TSingleMDP(world)
    # max_policies = single_mdp.get_all_policies("max")
    # policies = single_mdp.get_all_policies("exp")
    # env = BW4TCoopPOMDP(world, max_policies, policies)
    # # env = BW4TCoopPOMDP(world, policies, policies)
    # # env = BW4TCoopPOMDP(world, max_policies)
    # print(env.get_s((4, 10), (12, 1)))
    # env_set = BW4TCoopMDPSet(env, world, max_policies, task_graph)
    # pickle.dump((env, task_graph, env_set), open("results/bw4t/env.pkl", "wb"))
    # # print(env.get_s((4, 10), (12, 1)))
    # exit()

    # env, task_graph, env_set = pickle.load(open("tmp.pkl", "rb"))
    # # print(env.get_s((4, 10), (12, 1)))
    # # for t_id in range(6):
    # for t_id in range(len(task_graph.task_network)):
    #     print(t_id)
    #     env.set_world(env_set, task_graph, t_id)
    #
    #     b = make_belief(env.th)
    #     st = datetime.datetime.now()
    #     env.calc_a_vector(21, b)
    #     pickle.dump(env.a_vector_a, open(f"results/bw4t/policy_o_{t_id}.pkl", "wb"))
    #     # base_s = env.get_s((4, 10), (12, 1))
    #     check_s = env.get_s((4, 5), (12, 8))
    #     for a_r in range(env.a_r):
    #         # v = np.array(env.value_a(check_s, a_r, [0.25, 0.25, 0.25, 0.25]))
    #         print(a_r)
    #         print(env.a_vector_a[check_s][a_r])
    # exit()

    # env, task_graph, env_set = pickle.load(open("tmp.pkl", "rb"))
    # policy = pickle.load(open(f"results/bw4t/policy_5.pkl", "rb"))
    # s = env.get_s((4, 5), (12, 8))
    # for a_r in range(env.a_r):
    #     print(policy[s][a_r])
    # exit()

    env, task_graph, env_set = pickle.load(open("results/bw4t/env.pkl", "rb"))
    t_id = n_t_id = 0
    policies = []
    for t_id in range(len(task_graph.task_network)):
        # policies.append(pickle.load(open(f"results/bw4t/policy_o_{t_id}.pkl", "rb")))
        policies.append(pickle.load(open(f"results/bw4t/policy_{t_id}.pkl", "rb")))
    s = env.get_s((4, 10), (12, 1))

    policy = policies[t_id]
    task_net = task_graph.task_network[t_id]
    agent = Agent(env, env.r_policies, task_graph)
    agent.set_task(t_id)
    agent.set_policy(policy)
    agent.reset_belief()
    human = Human(env, env_set, env.r_policies, task_graph)
    human.set_task(t_id)
    count = 0
    while True:
        print(env.get_pos(s))
        goals = env.check_goal(s)
        for g in goals:
            if g in task_net:
                n_t_id = task_net[g]
        if n_t_id != t_id:
            t_id = n_t_id
            if t_id == -1:
                break
            policy = policies[t_id]
            agent.set_task(t_id)
            agent.set_policy(policy)
            agent.reset_belief()
            human.set_task(t_id)
            task_net = task_graph.task_network[t_id]
            if ("h", human.target) not in task_net:
                human.target = None
            print("---goal---")
        a_r = agent.act(s)
        t_s = env.get_next_s(s, a_r, None)
        human.update_belief(s, a_r)
        a_h = human.act(s)
        # agent.update_belief(s, a_h)
        # print(agent.belief)
        s = env.get_next_s(s, a_r, a_h)
        print(a_r, a_h)
        count += 1
    print(count)

