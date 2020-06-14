import os
import json
import datetime
from itertools import product

from scipy.special import softmax
import joblib
from joblib import Parallel, delayed
import numpy as np

from problem.bw4t.coop_mdp_set import BW4TCoopMDPSet
from problem.bw4t.coop_pomdp import BW4TCoopPOMDP
from problem.bw4t.task_graph import TaskGraph
from problem.bw4t.world import BW4T


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


def make_base_env(file_name, result_dir):
    world = BW4T()
    task_graph = TaskGraph(file_name)
    # world.print_world()
    base_env = BW4TCoopPOMDP(world, np.inf, 20)
    # base_env = BW4TCoopPOMDP(world, 20, 20)
    # print(base_env.get_s((4, 9), (8, 0)))
    # env_set = BW4TCoopMDPSet(base_env, world, np.inf, task_graph)
    env_set = BW4TCoopMDPSet(base_env, world, 20, task_graph)

    # joblib.dump(env_set.b_map, "test")
    # c = joblib.load("test")
    # for k, v in env_set.b_map.items():
    #     if not np.array_equal(v, c[k]):
    #         print(False)
    #         break
    # else:
    #     print(True)
    # exit()

    os.makedirs(f"{result_dir}/env/", exist_ok=True)
    joblib.dump((base_env, task_graph, env_set), f"{result_dir}/env/base_env.pkl")
    # joblic.dump()


def make_env(result_dir, t_ids=None):
    base_env, task_graph, env_set = joblib.load(f"{result_dir}/env/base_env.pkl")
    # _, task_graph_full, env_set_full = joblib.load(f"{result_dir}/env/base_env_full.pkl")
    if t_ids is None:
        t_ids = list(range(len(task_graph.task_network)))
    for t_id in t_ids:
        print(t_id)
        base_env.set_world(env_set, task_graph, t_id, (3,))
        joblib.dump(base_env, f"{result_dir}/env/env_{t_id}.pkl")


def policy_equal(policy, policy2):
    for s, a_r in product(range(len(policy)), range(len(policy[0]))):
        if not np.array_equal(policy[s][a_r], policy2[s][a_r]):
            print("false", s, a_r)
            print(policy[s][a_r], policy2[s][a_r])
            break
    else:
        print("true")


def make_policy(result_dir, policy_dir, t_id, algo):
    env = joblib.load(f"{result_dir}/env/env_{t_id}.pkl")
    b = make_belief(env.th)
    env.calc_a_vector(21, b, algo=algo)
    # c_test = joblib.load(f"{policy_dir}/{t_id}.pkl")
    # policy_equal(c_test, env.a_vector_a)
    joblib.dump(env.a_vector_a, f"{policy_dir}/{t_id}.pkl")



if __name__ == '__main__':
    policy_dir, algo = "base", 0
    # policy_dir, algo = "own", 1
    env_id = 2

    result_dir = f"results/bw4t/{env_id}"
    policy_dir = f"{result_dir}/policy_{policy_dir}/"

    make_base_env(f"problem/bw4t/map_data/map{env_id}.json", result_dir)
    # # make_env(result_dir)
    make_env(result_dir, [0])
    # exit()

    base_env, task_graph, env_set = joblib.load(f"{result_dir}/env/base_env.pkl")
    os.makedirs(f"{policy_dir}", exist_ok=True)
    # Parallel(n_jobs=-1, verbose=10)(
    #     [delayed(make_policy)(result_dir, policy_dir, t_id, algo) for t_id in
    #      range(len(task_graph.task_network))])
    make_policy(result_dir, policy_dir, 0, algo)
    # exit()


    base_env, task_graph, env_set = joblib.load(f"{result_dir}/env/base_env.pkl")
    t_id = 0
    policy = joblib.load(f"{policy_dir}/{t_id}.pkl")
    # check_s = base_env.get_s((4, 10), (8, 2))
    # check_s = base_env.get_s((5, 10), (8, 4))
    # check_s = base_env.get_s((7, 10), (9, 0))
    # check_s = base_env.get_s((9, 0), (7, 10))
    # check_s = base_env.get_s((5, 10), (8, 5))
    check_s = base_env.get_s((4, 10), (7, 0))

    # v = np.array([env_set.v_for_t_map_full[t_id][a][check_s] for a in task_graph.task_map[t_id].action_h])
    # b = softmax(v)
    print(check_s)
    b = np.array([1] * len(task_graph.task_map[t_id].action_h))
    b = b / np.sum(b)
    for a_r in range(base_env.a_r):
        print(policy[check_s][a_r])
        print(np.max(np.dot(policy[check_s][a_r], b)))
