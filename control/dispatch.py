# coding: utf-8
from copy import deepcopy
import numpy as np

def dispatch_sort(feature, priority):
    tmp = []
    for i in priority:
        tmp.append(feature[i])
    tmp = np.vstack((tmp, feature[len(priority):, :]))
    return tmp

def assign_sort(dispatch_out, priority):
    tmp = np.zeros(len(dispatch_out), dtype=np.int32)
    for i in range(len(priority)):
        tmp[priority[i]] = dispatch_out[i]
    return tmp

def get_dispatch_matrices(unit_num):
    dpt_out = []
    dpt_in = []

    for i in range(unit_num, 2, -1):
        for j in range(min(i, 5), -1, -1):
            if i + j <= 10:
                for k in range(min(j, 3), -1, -1):
                    if i + j + k <= 10 and 10 - i - j <= 2 * k:
                        dpt_out.append([i, j, k, 10 - i - j - k])
                        #print([i, j, k, 10 - i - j - k])
                    elif 10 - i - j > 2 * k:
                        break

    for i in range(unit_num, 1, -1):
        for j in range(min(i, 5), -1, -1):
            if i + j <= 10:
                for k in range(min(j, 3), -1, -1):
                    if i + j + k <= 10:
                        for t in range(min(k, 2), -1, -1):
                            if i + j + k + t <= 10 and 2 * t >= 10 - i - j - k:
                                dpt_in.append([i, j, k, t, 10 - i - j - k - t])
                                #print([i, j, k, t, 10 - i - j - k - t])
                            elif 2 * t < 10 - i - j - k:
                                break


    return dpt_out, dpt_in
