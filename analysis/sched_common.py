'''
This file includes all the data struct used in RTA
'''

from dataclasses import dataclass
from typing import List
from enum import IntEnum
import math

epsilon = 1 # = 1 for simulation experiments

# gpu execution segment, the execution time of each portion in one segment
@dataclass
class GPUseg:
    def __init__(self, G=0, Gm=0, Ge=0, Gd=0):
        self.G_hi = G
        self.Gm_hi = Gm
        self.Ge_hi = Ge
        self.Gd_hi = Gd
        self.G_lo = 0
        self.Gm_lo = 0
        self.Ge_lo = 0
        self.Gd_lo = 0

class CPUseg:
    def __init__(self, C=0):
        self.C_hi = C
        self.C_lo = 0

@dataclass
class Task:
    def __init__(self): 
        self.id = 0
        self.prio = 0 # prio == -1 for best effort tasks
        self.prio_gpu = self.prio
        self.C_hi = 0
        self.T = 0
        self.U = 0
        self.ges = []
        self.ces = []
        self.cpu = 0 # assigned to cpu 0 by default
        self.WR = 0
        self.hp_set = [] # a list of higher-priority task idx on CPU running in system
        self.hpp_set = [] # a list of higher-priority task idx on CPU running on the same core
        self.lp_set = []
        self.G_hi = 0
        self.Gm_hi = 0
        self.Ge_hi = 0
        self.Gd_hi = 0
        self.G_lo = 0
        self.Gm_lo = 0
        self.Ge_lo = 0
        self.Gd_lo = 0
        self.C_hi = 0
        self.C_lo = 0
        
    def apply_bounds(self, ratio = 1):
        # apply bound to each CPU segment
        for i in range(len(self.ces)):
            self.ces[i].C_lo = self.ces[i].C_hi * ratio
        # apply bound to cumulative CPU execution time
        self.C_lo = self.C_hi * ratio

        # apply bounds to each GPU segment
        for i in range(len(self.ges)):
            self.ges[i].G_lo = self.ges[i].G_hi * ratio
            self.ges[i].Gm_lo = self.ges[i].Gm_hi * ratio
            self.ges[i].Ge_lo = self.ges[i].Ge_hi * ratio
            self.ges[i].Gd_lo = self.ges[i].Gd_hi * ratio
        # compute the cumulative GPU execution time
        self.Gm_hi = 0
        self.Gd_hi = 0
        self.Ge_hi = 0
        for i in range(len(self.ges)):
            self.Gm_hi += self.ges[i].Gm_hi
            self.Gd_hi += self.ges[i].Gd_hi
            self.Ge_hi += self.ges[i].Ge_hi
        # apply bound to cumulative GPU execution time
        self.G_lo = self.G_hi * ratio
        self.Gm_lo = self.Gm_hi * ratio
        self.Ge_lo = self.Ge_hi * ratio
        self.Gd_lo = self.Gd_hi * ratio

def clear_task_response_time(tasks: List[Task]):
    for i in range(len(tasks)):
        tasks[i].WR = 0
    return tasks
