#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File name: main.py
# First Edit: 2021-09-06
# Last Change: 2021-09-06
import pickle
from typing import Any
from typing import List
from typing import NamedTuple
from typing import NewType
from typing import Optional
from typing import Set
from typing import Union

import numpy as np
import pandas as pd
import platypus
from nptyping import Int8
from nptyping import NDArray
from numpy.random import Generator
from numpy.random import PCG64
from platypus import NSGAII
from platypus import Problem
from platypus import ProcessPoolEvaluator
from platypus import Real
from typing_extensions import Literal

side_name = "Quaternary_個人_横_tempo"
vert_name = "Quaternary_通常_縦"
scedule_name = "Primary_予定(202106)"
file_name = "学園勤務ルール_現在編集中.xlsx"
workmethod_name = "Primary_勤務"
relation_name = "Primary_関係"

side_df = pd.read_excel(
    file_name, sheet_name=side_name, header=[0, 1, 2, 3], index_col=[0, 1, 2]
).applymap(lambda x: x if isinstance(x, (int, float)) else np.nan).index.droplevel()

scedule_df = pd.read_excel(
    file_name, sheet_name=scedule_name, header=[0], index_col=[0]
)

class Worker:
    def __init__(self, value):
        self.name = value["名前(固定)"]
        self.floor = value["勤務階"]
        self.workplace = value["職場"]
        self.role = value["役割"]
        self.exception = value["例外ルール"]
        self.data = ""


workmethod_df = pd.read_excel(
    file_name, sheet_name=workmethod_name, header=[0], index_col=[0]
)


workmethod_df.columns = [
    "long_name",
    "short_name",
    "vert_name",
    "side_name",
    "question",
    "target_value",
]
side_workmethod_df = workmethod_df[["side_name", "target_value"]].dropna()
side_workmethod_df.target_value = side_workmethod_df.target_value.apply(
    lambda x: x if not isinstance(x, (str)) else list(map(int, x.rsplit(",")))
)

with open("workers.pkl", "rb") as f:
    workers = pickle.load(f)


def calc_penalty(count, condition, big_penalty=1000, small_penalty=50):
    if condition["固定値"]:
        return abs(count - condition["固定値"]) * big_penalty

    if condition["最小値"] and condition["最小値"] > count:
        return abs(count - condition["最小値"]) * big_penalty

    if condition["最大値"] and count > condition["最大値"]:
        return abs(count - condition["最大値"]) * big_penalty

    if condition["希望値"]:
        if condition["重み"]:
            weight = condition["重み"]
        else:
            weight = small_penalty

        return abs(condition["希望値"] - count) * weight
    else:
        return 0


def calc_scedule_penalty_of_a_worker(
    worker_scedule, worker_condition, side_work_methods, debug=False
):
    s = worker_scedule

    target = len(s[s > 0])
    penalty = calc_penalty(target, worker_condition["基本条件"]["出勤回数"]["総計"])

    for i in side_work_methods.values:
        if isinstance(i[0], (float, int)):

            target = len(s[s == i[0]])
        else:
            target = len([sx for sx in s if sx in i[0]])
        new_penalty = calc_penalty(target, worker_condition["基本条件"]["出勤回数"][i[1]])
        penalty += new_penalty

    return penalty


days = len(scedule_df.T)
worker_amount = len(side_df)
side_workmethods = workmethod_df[["target_value", "side_name"]].dropna()
side_workmethods.target_value = side_workmethods.target_value.apply(
    lambda xx: xx if not isinstance(xx, str) else list(map(int, xx.rsplit(",")))
)
side_workmethods.target_value = side_workmethods.target_value.apply(
    lambda xx: int(xx) if isinstance(xx, (float)) else xx
)

variables = days * worker_amount

def schaffer(x):
    penalty = 0
    # print(x)
    x = np.array(x, dtype=np.int8)

    for ind, worker in enumerate(workers):
        penalty += calc_scedule_penalty_of_a_worker(
            x[ind * days : days * (ind + 1)], worker.data, side_workmethods
        )

    xx = x.reshape(worker_amount, days)

    for d in range(days):
        data = xx[:, d]
        penalty += calc_vert_penalty(data, side_df, vert_workmethods)

    return penalty


problem = Problem(variables, 1)
problem.directions[:] = Problem.MINIMIZE
problem.types[:] = [platypus.Integer(0, 7)] * variables
problem.function = schaffer

with ProcessPoolEvaluator(8) as evaluator:
    algorithm = NSGAII(problem, evaluator=evaluator)
    algorithm.run(500)

int8 = platypus.Integer(0, 10)

x = np.array(
    [int8.decode(i) for i in solution.variables],
    dtype=np.int8,
)

data = x.reshape(worker_amount, days)
jp_name_dict = {0: "休み", 1: "普", 2: "早", 3: "遅", 4: "遅R", 5: "P", 6: "当1", 7: "当2", 8: "明1", 9: "明2", 10: "普"}

df = pd.DataFrame(
    data, index=side_df.index, columns=scedule_df.columns
).applymap(jp_name_dict.get)
df.to_excel('output.xlsx')

# vim: ft=python ts=4 sw=4 sts=4 tw=88 fenc=utf-8 ff=unix si et:
