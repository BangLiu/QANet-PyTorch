# -*- coding: utf-8 -*-
import numpy as np


def common(list1, list2):
    return list(set(list1).intersection(list2))


def substract(list1, list2):
    return list(set(list1) - set(list2))


def remove_values_from_list(l, val):
    return [value for value in l if value != val]


def del_list_indexes(l, id_to_del):
    somelist = [i for j, i in enumerate(l) if j not in id_to_del]
    return somelist


def del_list_inplace(l, id_to_del):
    for i in sorted(id_to_del, reverse=True):
        del(l[i])


def del_list_numpy(l, id_to_del):
    arr = np.array(l, dtype='int32')
    return list(np.delete(arr, id_to_del))
