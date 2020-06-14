import os
import joblib
import json
from problem.bw4t.no_human_agent import Agent
from algo.policy_util import make_poilcy
import numpy as np


def dump_env_data_for_js(base_env, task_graph, env_set, result_dir):
    task_info = task_graph.make_task_net_for_js()
    env_info = base_env.make_base_info_for_js()
    best_goals = env_set.make_best_goals_for_js()
    greedy_policies = base_env.greedy_policies
    os.makedirs(f"{result_dir}/env/", exist_ok=True)
    json.dump({"env": env_info, "task": task_info,
               "greedy": greedy_policies, "best_goals": best_goals},
              open(f"{result_dir}/env/env_for_js.json", "w"))

def dump_policy_for_js(policy, i_pi_h, task_a_r, task_a_h, file_name):
    policy_json = {s: {a_r: p.tolist() for a_r, p in a_p.items()} for s, a_p in policy.items()}
    i_pi_h_json = {s: {a_h: i_pi_h[s][a_h].tolist() for a_h in range(len(i_pi_h[s]))}
                   for s in range(len(i_pi_h))}
    t_a_r = [t[1] for t in task_a_r]
    t_a_h = [t[1] for t in task_a_h]
    json.dump({"policy": policy_json, "i_pi_h": i_pi_h_json, "task_a_r":t_a_r, "task_a_h":t_a_h},
              open(file_name, "w"))

if __name__ == '__main__':
    policy_dir, algo = "base", 0
    # policy_dir, algo = "own", 1

    result_dir = "results/bw4t/5"
    policy_dir = f"{result_dir}/policy_{policy_dir}/"

    base_env, task_graph, env_set = joblib.load(f"{result_dir}/env/base_env.pkl")

    dump_env_data_for_js(base_env, task_graph, env_set, result_dir)
    exit()

    agent = Agent(base_env, env_set, policy_dir, 1)
    os.makedirs(f"{policy_dir}/js/", exist_ok=True)
    for t_id in list(range(len(task_graph.task_network))):
        print(t_id)
        policy = joblib.load(f"{policy_dir}/{t_id}.pkl")
        agent.set_task(task_graph.task_map[t_id], 0, True)
        dump_policy_for_js(policy, agent.i_pi_h, agent.task.action_r, agent.task.action_h,
                           f"{policy_dir}/js/{t_id}.json")
