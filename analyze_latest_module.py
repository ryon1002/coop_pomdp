import numpy as np
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
# from matplotlib import rcParams
import itertools

# plt.rcParams["font.family"] = "MS ゴシック"
import pandas as pd

# color_map_4 = {0: "red", 1: "green", 2: "blue", 3: "grey"}
# label_map_4 = {0: "Ours(with Predictability Bias)", 1: "Optimistic CIRL",
#                2: "Simple POMDP", 3: "Explicit"}

color_map_3 = {0: "red", 1: "blue", 2: "green"}
label_map_3 = {0: "Ours", 1: "CIRL", 2: "Simple POMDP"}

color_map_3_2 = {0: "red", 1: "yellow", 2: "blue"}
label_map_3_2 = {0: "Implicit Guidance", 1: "Explicit Guidance", 2: "No Guidance"}
# label_map_3_2 = {0: "暗黙的な行動誘導", 1: "明示的な行動誘導", 2: "行動誘導なし"}


def make_evaluate_data(data):
    q = np.array([data["q"][i] for i in ["0", "1", "2"]]).astype(np.int)
    return q


def check_valid_user_1(q):
    return np.sum(q == 4) != 12


def check_valid_user_1(q):
    return np.sum(q == 4) != 12


cc_count = 0
def extract_dd(data, i, answer):
    global cc_count
    dist, done = int(data[0]), int(data[-1][-1])
    if int(i) == 60:
        cc_count += 1
        if cc_count % 2:
            dist, done = 17, 1

    if done == 0 or done == 10:
        dist = dist + 3
    if done == 1 or done == 11:
        dist = dist + 1
    return dist, done, done == answer


def check_valid_user_2(data, indexes_c, answer):
    miss = 0
    for i in indexes_c:
        dist, done, correct = extract_dd(data["r"][i], i, answer[i])
        if done == 10:
            miss += 1
    # print(miss)
    # return True
    return miss < 2

def check_valid_user_3(data, indexes_c):
    miss = 0
    for i in indexes_c:
        dist, done, correct = extract_dd(data["r"][i], i, 0)
        if done == -1:
            miss += 1
            continue
        if i == "60" and done >= 10:
            miss += 1
    # print(miss)
    # return True
    return miss < 1
    exit()
    return miss < 2

def analyze_dist(data, indexes):
    data = [np.array([d[0] for d in data[i] if d[1] in [0, 1]]) for i in indexes]
    mean = [np.mean(d) for d in data]
    p = {(i, j): scipy.stats.ttest_ind(data[i], data[j], equal_var=False)[1] for i, j in
         itertools.combinations(range(3), 2)}
    print(mean, p)
    # plot_bar(mean, p, (15, 27), color_map_3, label_map_3)
    # plt.savefig("result_3.eps", bbox_inches="tight", format="eps")
    # plt.show()


def analyse_user_result(c_list):
    base_idx = [210, 220, 230, 310, 320]
    base_idx_1 = [210, 220, 230]
    base_idx_2 = [310, 320]
    c_all_list = np.array([np.array([c_list[bi + i] for bi in base_idx]).flatten()
                           for i in range(3)], np.int).T
    c_all_list_1 = np.array([np.array([c_list[bi + i] for bi in base_idx_1]).flatten()
                             for i in range(3)], np.int).T
    c_all_list_2 = np.array([np.array([c_list[bi + i] for bi in base_idx_2]).flatten()
                             for i in range(3)], np.int).T
    # c_list = {bi: np.array([c_list[bi + i] for i in range(3)], np.int).T for bi in base_idx}
    # print(matplotlib.__version__)
    # exit()
    mean, p = get_user_result(c_all_list[:, [0, 2, 1]])
    for j in range(len(color_map_3_2)):
        plt.bar(0 + (0.2 * (j) - 0.2), mean[j], width=0.2,
                color=color_map_3_2[j], label=label_map_3_2[j])
    add_sd_score([0 + (0.2 * (j - 1)) for j in range(3)], mean, p)
    mean, p = get_user_result(c_all_list_1[:, [0, 2, 1]])
    for j in range(len(color_map_3_2)):
        plt.bar(1 + (0.2 * (j) - 0.2), mean[j], width=0.2,
                color=color_map_3_2[j])
    add_sd_score([1 + (0.2 * (j - 1)) for j in range(3)], mean, p)
    mean, p = get_user_result(c_all_list_2[:, [0, 2, 1]])
    for j in range(len(color_map_3_2)):
        plt.bar(2 + (0.2 * (j) - 0.2), mean[j], width=0.2,
                color=color_map_3_2[j])
    add_sd_score([2 + (0.2 * (j - 1)) for j in range(3)], mean, p)
    # anova_1(c_all_list_1, c_all_list_2)
    # from scipy import stats
    # result = stats.f_oneway(c_all_list[0], c_all_list[1], c_all_list[2])
    # print(result)
    # exit()
    plt.ylim(0, 1.5)
    plt.legend(loc="upper right", fontsize=14)
    # plt.legend(loc="upper right", fontsize=19, prop={"family":"Meiryo"})
    # plt.legend(loc="upper right", prop={"size":14, "family":"Meiryo"})
    plt.xticks([0, 1, 2], ["All", "(A)", "(B)"])
    plt.xlabel("Task type", fontsize=14)
    # plt.xlabel("環境の種類", fontsize=14, fontname="Meiryo")
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.ylabel("Average rate of capture the best object", fontsize=14)
    # plt.ylabel("タスク成功率", fontsize=14, fontname="Meiryo")
    # plt.ylabel("TT", fontsize=14, fontname="MS Gothic")
    # plt.savefig("result_1.pdf", bbox_inches="tight", format="pdf")
    plt.savefig("result_1.eps", bbox_inches="tight", format="eps")
    plt.show()
    # print(get_user_result(c_all_list))

import statsmodels.stats.anova as anova
from statsmodels.graphics.factorplots import interaction_plot
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

def get_list(j):
    ret = [0, 0, 0]
    ret[j] = 1
    return tuple(ret)

def anova_1(cl1, cl2):
    data = []
    for i in cl1:
        for j, s in enumerate(i):
            # data.append((1, 0) + get_list(j) + (s,))
            data.append((0, j, s))
    for i in cl2:
        for j, s in enumerate(i):
            # data.append((0, 1) + get_list(j) + (s,))
            data.append((1, j, s))
    data = pd.DataFrame(data, columns=["env", "agent", "score"])
    # data = pd.DataFrame(data, columns=["env1", "env2",
    #                                    "agent1", "agent2", "agent3", "score"])

    # for h, i in enumerate(cl1):
    #     for j, s in enumerate(i):
    #         data.append((h, 0, j, s))
    # for h, i in enumerate(cl2):
    #     for j, s in enumerate(i):
    #         data.append((h, 1, j, s))
    # data = pd.DataFrame(data, columns=["subject", "env", "agent", "score"])
    # aov=anova.AnovaRM(data, 'score', "subject", ["env", "agent"])
    # print(aov)


    # formula = "score ~ C(env) + C(agent) + C(env):(agent)"
    formula = "score ~ C(env) * C(agent)"
    # formula = "score ~ (agent1) + (agent2) + (agent3)"
    # formula = "score ~ (agent)"
    model = ols(formula, data).fit()
    # aov_table = anova_lm(model, typ=1)
    aov_table = anova_lm(model, typ=2)
    print(aov_table)
    exit()

def anova_2(d):
    data = []
    for i in d:
        for j, s in enumerate(i):
            data.append((j, s))
    data = pd.DataFrame(data, columns=["agent", "score"])
    formula = "score ~ C(agent)"
    model = ols(formula, data).fit()
    aov_table = anova_lm(model, typ=2)
    print(aov_table)
    # exit()

def get_user_result(data):
    mean = np.mean(data, 0)
    p = {(i, j): scipy.stats.ttest_rel(data[:, i], data[:, j])[1] for i, j in
         itertools.combinations(range(3), 2)}
    return mean, p


def analyse_user_evaluate(data):
    mean = np.mean(data, 0)
    p = {(i, j): scipy.stats.wilcoxon(data[:, i], data[:, j])[1] for i, j in
         itertools.combinations(range(3), 2)}
    return mean, p


def plot_bar(data, p, y_limit, color_map, label_map):
    for i in range(len(color_map)):
        plt.bar(i, data[i], color=color_map[i], label=label_map[i])

    p_h_space = 0.5
    p_pos = np.max(data) + p_h_space
    pk_list = sorted(p.keys(), key=lambda x: abs(x[0] - x[1]))
    for pk in pk_list:
        if p[pk] < 0.03:
            plt.plot([pk[0], pk[1]], [p_pos, p_pos], color="k")
            plt.plot([pk[0], pk[0]], [p_pos, p_pos - (p_h_space * 0.6)], color="k")
            plt.plot([pk[1], pk[1]], [p_pos, p_pos - (p_h_space * 0.6)], color="k")
            c = "**" if p[pk] < 0.01 else "*"
            plt.text(float(pk[0] + pk[1]) / 2, p_pos - (p_h_space * 0.6), c)
            p_pos += p_h_space
    plt.legend(loc="upper right")
    plt.ylim(*y_limit)
    plt.xticks([], [])


def analyse_user_allevaluate(q_list):
    means = []
    p_s = []
    for check in range(4):
        data = q_list[:, :, check]
        anova_2(data)
        m, p = analyse_user_evaluate(data)
        means += list(m)
        p_s.append(p)
    # exit()
    means = np.array(means)
    for i in range(1):
        for j in range(len(color_map_3_2)):
            plt.bar(i + (0.25 * (j - 1)), means[i * 3 + j], width=0.25,
                    color=color_map_3_2[j], label=label_map_3_2[j])
        add_sd_eval([i + (0.25 * (j - 1)) for j in range(3)],
                   means[i * 3:i * 3 + 3], p_s[i])

    for i in range(1, 4):
        for j in range(len(color_map_3_2)):
            plt.bar(i + (0.25 * (j - 1)), means[i * 3 + j], width=0.25,
                    color=color_map_3_2[j])
        add_sd_eval([i + (0.25 * (j - 1)) for j in range(3)],
                    means[i * 3:i * 3 + 3], p_s[i])
    plt.legend(loc="upper right", fontsize=14)
    plt.ylim(0, 11)
    plt.xticks([0, 1, 2, 3], ["(1)", "(2)", "(3)", "(4)"])
    plt.xlabel("Survey Item", fontsize=14)
    plt.yticks(range(1, 8), range(1, 8))
    plt.ylabel("Average survey score", fontsize=14)
    plt.savefig("result_2.eps", bbox_inches="tight", format="eps")
    # plt.savefig("result_2.pdf", bbox_inches="tight", format="pdf")
    plt.show()

def add_sd_score(x, y, p):
    y_diff = 0.02
    x_diff = 0.04
    add_one_sd(x[0] + x_diff, x[1] - x_diff, 0.85,
               y[0] + y_diff, y[1] + y_diff, p[(0, 1)])
    add_one_sd(x[1] + x_diff, x[2] - x_diff, 0.85,
               y[1] + y_diff, y[2] + y_diff, p[(1, 2)])
    add_one_sd(x[0] - x_diff, x[2] + x_diff, 0.95,
               y[0] + y_diff, y[2] + y_diff, p[(0, 2)])

def add_sd_eval(x, y, p):
    y_diff = 0.1
    x_diff = 0.04
    add_one_sd(x[0] + x_diff, x[1] - x_diff, 6.5,
               y[0] + y_diff, y[1] + y_diff, p[(0, 1)])
    add_one_sd(x[1] + x_diff, x[2] - x_diff, 6.5,
               y[1] + y_diff, y[2] + y_diff, p[(1, 2)])
    add_one_sd(x[0] - x_diff, x[2] + x_diff, 7,
               y[0] + y_diff, y[2] + y_diff, p[(0, 2)])

def add_one_sd(lx, rx, uy, lyl, lyr, p):
    if p < 0.03:
        plt.plot([lx, rx], [uy, uy], color="k")
        plt.plot([lx, lx], [uy, lyl], color="k")
        plt.plot([rx, rx], [uy, lyr], color="k")
        c = "**" if p < 0.01 else "*"
        plt.text(float(lx + rx) / 2 - 0.06, uy + 0.02, c, fontsize=14)
