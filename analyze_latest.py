import csv, json
from collections import defaultdict

import numpy as np
import analyze_latest_module

# reader = csv.reader(open("analize_result/result.txt"), delimiter="\t")
# indexes_v = ['60', '61', '63']
# answer = {"60":0, "61":0, "63":0}

reader = csv.reader(open("analize_result/result_2.txt"), delimiter="\t")
indexes_v = ['210', '211', '212', '220', '221', '222', '230', '231', '232',
             '310', '311', '312', '320', '321', '322']
answer = {'210': 1, '211': 1, '212': 1, '220': 0, '221': 0, '222': 0,
          '230': 0, '231': 0, '232': 0, '310': 1, '311': 1, '312': 1,
          '320': 0, '321': 0, '322': 0, '240': 0, '241': 0}

q_list = []
r_dist_list = {int(i): [] for i in indexes_v}
r_list = {int(i): [] for i in indexes_v}
c_list = {int(i): [] for i in indexes_v}
# indexes = ['20', '21', '22', '23', '30', '31', '32', '33',
c = 0
for l in reader:
    data = json.loads("{\"" + l[1][1:-1])
    q = analyze_latest_module.make_evaluate_data(data)
    if not analyze_latest_module.check_valid_user_1(q):
        # print("filter")
        continue

    # if not analyze_latest_module.check_valid_user_2(data, indexes_c, answer):
    #     print("filter")
    #     continue

    # if not analyze_latest_module.check_valid_user_3(data, indexes_v):
    #     continue
    c += 1

    q_list.append(q)
    for i in indexes_v:
        dist, done, correct = analyze_latest_module.extract_dd(data["r"][i], i, answer[i])
        r_dist_list[int(i)].append(done)
        r_list[int(i)].append((dist, done))
        c_list[int(i)].append(correct)
print(c)
# print(r_list)

# print(r_list)
# exit
analyze_latest_module.analyse_user_result(c_list)
exit()

for i in indexes_v:
    check_dict = defaultdict(int)
    for j in r_list[int(i)]:
        check_dict[j] += 1
    print(i, check_dict)
# exit()

# point!!
print(len(r_list))
# analyze_latest_module.analyze_dist(r_list, [60, 61, 63])

# point!!
q_list = np.array(q_list)
q_list = q_list[:, [0, 2, 1]]
analyze_latest_module.analyse_user_allevaluate(q_list)
