# coding: utf-8

import numpy as np
import math

def list2str(l, split_char=','):
    if type(l) == int:
        l = [l]
    s = split_char.join([str(i) for i in l])
    return s

def dict2str(d, split_char='|', equal_char=':'):
    l = []
    for (key, values) in d.items():
        l.append('%s:%s' % (key, value))

    s = split_char.join(l)
    return s

def str2list(s):
    s = s.split('=')[-1]
    l = [float(i) for i in s.split(',')]
    return l

def str2dict(s, split_char='|', equal_char=':', left_type=None):
    '''
    left_type:id_list时需将key转为int类型
    '''
    if s[-1] == '|':
        s = s[:-1]
    d = dict()
    ss = s.split('=')
    for pair in ss[-1].split(split_char):
        (key, value) = pair.split(equal_char)
        if 'int' == left_type:
            key = int(key)
        if value.find(',') > -1:
            value = str2list(value)
        else:
            try:
                value = float(value)
            except:
                pass
        d[key] = value

    return ss[0], d

def get_simple_id_list(l):
    '''
    将一个常规id_list转化为简写的str
    '''
    l = list(l)
    l.sort()
    s = ''
    st, ed = 0, 0
    while ed < len(l):
        while ed + 1 < len(l) and ed - st + 1 == l[ed + 1] - l[st]:
            ed += 1
        if st == ed:
            s += '%d,' % l[st]
        else:
            s += '%d-%d,' % (l[st], l[ed])
        ed += 1
        st = ed
    return s[:-1]

def simple_list_to_full(s):
    '''
    将间歇的id_list转为常规list
    输入:s  str
    return list
    '''
    l = s.split(',')
    result = []
    for uid in l:
        if uid.find('-') > -1:
            (st, ed) = uid.split('-')
            for i in range(int(st), int(ed) + 1):
                result.append(i)
        else:
            result.append(int(uid))

    return result

def get_ai(d):
    ##############################
    #s='<%s/>'%''
    return ''

def get_action(d):
    '''
    动作字典转为指令
    '''
    for (key, value) in d.items():
        if 'objid_list' == key:
            value = get_simple_id_list(value)
        elif 'ai' == key:
            value = get_ai(value)
        else:
            value = '[%s]' % list2str(value, split_char='|')


def pos2mapid(pos):
    '''
    真实坐标转5*5地图坐标
    '''
    x = (pos[0] + 275) // 50
    x = min(4, max(0, x))
    y = (pos[2] + 120) // 60
    y = min(4, max(0, y))
    return int(x), int(y)

def get_dis(pos1, pos2, including_h=False):
    '''
    计算两点间距离
    '''
    dis = (pos1[0] - pos2[0]) ** 2 + (pos1[2] - pos2[2]) ** 2
    if including_h:
        dis += (pos1[1] - pos2[1]) ** 2
    return dis ** 0.5