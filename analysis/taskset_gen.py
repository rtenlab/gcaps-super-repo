"""
This file is used for random taskset generation

Only function `taskset_generation()` as well as its return value needs to be exposed.
"""

import numpy as np
import random
import time
import inspect
from enum import IntEnum
from termcolor import colored

from sched_common import *


class UtilMode(IntEnum):
    regular = 0
    light = 1
    heavy = 2

@dataclass
class CPU:
    def __init__(self):
        self.id = 0
        self.util = 0
        self.pin1 = 0
        self.pin2 = 0

def get_rand(val_range, step=0):
    minval = val_range[0]
    maxval = val_range[1]

    if minval > maxval:
        return -1
    if minval == maxval:
        return maxval
    
    if isinstance(minval, int) and isinstance(maxval, int):
        return random.randint(minval, maxval)
    		
    if step == 0:
        return round(random.randint(int(minval * 100), int(maxval * 100)) / 100, 2)
    else:
        return round(random.randrange(int(minval * 100), int(maxval * 100), step * 100) / 100, 2)
    
# get `n` unique numbers from the range of [`start`, `end`]. return a list
def get_rand_set(start, end, n):
    if end - start + 1 < n:
        print("Error! Invalid input to {}: start {}, end {}, n {}".format(inspect.currentframe().f_code.co_name, start, end, n))
        exit()
    random.seed(time.process_time()) 
    numbers = set()
    while len(numbers) < n:
        numbers.add(random.randint(start, end))

    return list(numbers)
    

# randomly divide `total` into `n` segments, by returning a list with size of `n`
# it uses UuniFast
# note that 0 might be generated
def random_divider(total: float, n: int) -> List[float]:
    if total == 0:
        return [0] * n
    
    random.seed(time.process_time()) 
    segments = []
    rem_total = total
    for i in range(1, n):
        # rand_num = random.uniform(0, 1) # yidi: first version, now change to uunifast
        rand_num = random.random() ** (1.0 / (n - i))
        seg = rand_num * rem_total
        segments.append(round(seg, 2))
        rem_total -= seg
    segments.append(round(rem_total, 2))
    random.shuffle(segments)
    return segments 


def sort_tasks_RM(tasks: List[Task]):
    rt_tasks = sorted([task for task in tasks if task.prio >= 0], key=lambda task: task.T)
    be_tasks = sorted([task for task in tasks if task.prio == -1], key=lambda task: task.id)
    return  rt_tasks + be_tasks


def sort_tasks_util(tasks: List[Task], decreasing=True):
    return sorted(tasks, key=lambda task: task.U, reverse=decreasing)


# False for WFD, True for BFD
def sort_cpus_util(cpus: List[CPU], decreasing=False):
    return sorted(cpus, key=lambda cpu: cpu.util, reverse=decreasing)


##########################################################
# Parameters for taskset generation
##########################################################
@dataclass
class Params():
    def __init__(self):
        self.ncpu = 4
        self.range_num_tasks_per_cpu = [3, 6]
        self.range_ratio_of_gpu_tasks = [0.4, 0.6] 
        self.range_task_period = [[30, 500], [30, 500]]
        self.range_util_per_cpu = [0.4, 0.6] 
        self.range_task_util = [0.05, 0.3]
        self.range_num_ges_per_task = [1, 3]
        self.range_ratio_g_to_c = [0.2, 2]
        self.range_ratio_gm_to_g = [0.1, 0.3]
        self.range_ratio_gd_to_ge = [0.2, 0.8]
        self.range_ratio_lo_to_hi = [0.5, 1.0] # ratio of lower bound to upper bound
    

##############################################################################
# @var `util_dist_mode` - 0: uniform, 1: bimodal_light, 2: bimodal_heavy
# @param `is_pinned_task` - whether this is the task with small period but high utilization, which is supposed to be pinned to each cpu

# @return a single task with valid task parameters
##############################################################################
def create_single_task(params: Params, 
                       expr_gid: int,
                    is_gpu_task: bool = True, 
                    pinned_task_type: int = -1,
                    cpuid: int = -1) -> Task:
    random.seed(time.process_time())

    task = Task()
    task.prio = 0 # temporarily
    task.cpu = cpuid # -1 means unassigned

    if expr_gid == 1: # regular taskset generation for regular gcaps rta experiment
        task.T = get_rand(params.range_task_period[0], 10)
        task.U = get_rand(params.range_task_util, 0.05)
    elif expr_gid == 2: # Modified taskset generation for improved gcaps analysis
        if pinned_task_type == 0: # small period
            task.T = get_rand(params.range_task_period[0], 10)
            task.U = get_rand([0.5, 0.7])
        elif pinned_task_type == 1: # large period
            task.T = get_rand(params.range_task_period[1], 10)
            task.U = get_rand([0.3, 0.5])
        else: # whatever
            task.T = get_rand([100, 200], 10)
            task.U = get_rand([0.05, 0.3])
    
    E = task.U * task.T

    n_ges = 0
    if is_gpu_task == True:
        task.C_hi = round(E / (1 + get_rand(params.range_ratio_g_to_c)), 2)
        task.G_hi = round(E - task.C_hi, 2)    
        n_ges = get_rand(params.range_num_ges_per_task, 0)
        ges = random_divider(task.G_hi, n_ges)
        for i in range(n_ges):
            task.ges.append(GPUseg())
            task.ges[i].G_hi = ges[i]
            task.ges[i].Gm_hi = ges[i] * get_rand(params.range_ratio_gm_to_g, 0)
            task.ges[i].Ge_hi = ges[i] - task.ges[i].Gm_hi
    else: # cpu exec only
        task.C_hi = E
        task.G_hi = 0

    n_ces = 0
    if is_gpu_task == True:
        n_ces = n_ges + 1
    else:
        n_ces = 1
    ces = random_divider(task.C_hi, n_ces)
    for i in range(n_ces):
        task.ces.append(CPUseg())
        task.ces[i].C_hi = ces[i]
    
    task.apply_bounds(get_rand(params.range_ratio_lo_to_hi, 0)) 
    return task

# note: taskset generation approach for the first version of expr
def random_taskset(params: Params, expr_gid: int, num_tasks: int, percentage_be_tasks: float):
    random.seed(time.process_time())

    tasks = [] # all the tasks
    if num_tasks == -1: # meaning that the number is decided by ncpu
        num_tasks = get_rand(params.range_num_tasks_per_cpu, 0) * params.ncpu

    if expr_gid == 1: # general sim or gpu prio assign
        for i in range(num_tasks):
            is_gpu_task = random.random() < get_rand(params.range_ratio_of_gpu_tasks, 0)
            task = create_single_task(params, expr_gid, is_gpu_task, -1)
            tasks.append(task)
    elif expr_gid == 2: # improved analysis
        for i in range(num_tasks):
            # for each cpu, at least two tasks, one cpu-intensive with small period and one gpu-intensive with large period
            is_gpu_task = False
            pinned_task_type = -1
            cid = -1
            if i < min(2, params.ncpu): 
                pinned_task_type = 0
                is_gpu_task = False
                cid = i
            elif i < min(3, params.ncpu + 1): 
                pinned_task_type = 1
                is_gpu_task = True
                cid = i - 2
            else:
                is_gpu_task = random.random() < get_rand(params.range_ratio_of_gpu_tasks, 0)
            task = create_single_task(params, expr_gid, is_gpu_task, pinned_task_type, cid)
            tasks.append(task)

        # * assign priority globally
    if percentage_be_tasks > 0:
        # randomly select number of best-effort tasks
        num_be_tasks = int(percentage_be_tasks * num_tasks)
        idxs = get_rand_set(0, len(tasks)-1, num_be_tasks)
        for idx in idxs:
            tasks[idx].prio = -1 # set to be best-effort task
    return tasks

#####################################################################
# Generate a list of tasks with uunifast
#####################################################################
def random_taskset_uunifast(params: Params, expr_gid: int, num_tasks: int, percentage_be_tasks: float):
    random.seed(time.process_time())

    tasks = []
    # here we decide for each cpu first
    if expr_gid == 1: # general sim or gpu prio assign
        if num_tasks <= 0:
            for cpuid in range(params.ncpu):
                num_tasks_on_cpu = get_rand(params.range_num_tasks_per_cpu)
                util_cpu = get_rand(params.range_util_per_cpu)
                util_list = random_divider(util_cpu, num_tasks_on_cpu)
                for j in range(num_tasks_on_cpu):
                    is_gpu_task = random.random() < get_rand(params.range_ratio_of_gpu_tasks, 0)
                    params.range_task_util[0] = params.range_task_util[1] = max(util_list[j], 0.01) # forcing the generated util is non-zero
                    task = create_single_task(params, expr_gid, is_gpu_task, -1, cpuid)
                    tasks.append(task)
        elif num_tasks > 0:
            # generate util for each task, then follow wfd
            for i in range(num_tasks):
                is_gpu_task = random.random() < get_rand(params.range_ratio_of_gpu_tasks, 0)
                task = create_single_task(params, expr_gid, is_gpu_task, -1, -1)
                tasks.append(task)

    num_tasks = len(tasks)
    # * assign priority globally
    if percentage_be_tasks > 0:
        # randomly select number of best-effort tasks
        num_be_tasks = int(percentage_be_tasks * num_tasks)
        idxs = get_rand_set(0, len(tasks)-1, num_be_tasks)
        for idx in idxs:
            tasks[idx].prio = -1 # set to be best-effort task
    return tasks        


#####################################################################
# Task generation 
# only this function needs to be exposed to experiments

# After a taskset is generated by `random_taskset_uunifast()`, load balancing is done in the current function based on WFD.

# @param `ncpu` - number of cores, by default = 3
# @param `num_heavy_tasks` - 
# @param `num_be_tasks` - average number of be tasks per core
# @return the list of tasks sorted with RM, with all parameters ready
# Tasks have global priorities
#####################################################################
def taskset_generation(params: Params, expr_gid: int, num_tasks: int = -1, percentage_be_tasks: int = 0) -> List[Task]:
    random.seed(time.process_time())

    # fixme： temporary for expr gid 2
    if expr_gid == 1 or expr_gid == 3:
        tasks = random_taskset_uunifast(params, expr_gid, num_tasks, percentage_be_tasks)
    elif expr_gid == 2:
        tasks = random_taskset(params, expr_gid, num_tasks, percentage_be_tasks)
    
    # tasks = manual_taskset()

    # * Task allocation
    cpus = []
    for i in range(params.ncpu):
        cpus.append(CPU())
        cpus[i].id = i
    # init with pre-assigned tasks
    for i in range(len(tasks)):
        cid = tasks[i].cpu
        if cid != -1:
            cpus[cid].util += tasks[i].U
    sort_tasks_util(tasks) # sort with decreasing util first
    for i in range(len(tasks)):
        if tasks[i].cpu != -1: # already assigned
            continue
        # WFD
        cpus = sort_cpus_util(cpus)
        cpus[0].util += tasks[i].U
        tasks[i].cpu = cpus[0].id

    # * sort the tasks with RM, rt tasks first. And assign task id accordingly
    tasks = sort_tasks_RM(tasks)
    for i in range(len(tasks)):
        if tasks[i].prio == -1: # do not assign for be tasks
            continue
        else:
            num_be_tasks = int(len(tasks) * percentage_be_tasks)
            tasks[i].prio = len(tasks) - num_be_tasks - i # the larger the higher
        tasks[i].id = i
    # * get list of higher-priority tasks, hpp and hp
    for i in range(len(tasks)):
        if tasks[i].prio == -1:
            break # tasks alreay sorted, no need to record this info for be tasks
        for h in range(len(tasks[:i])):
            tasks[i].hp_set.append(h)
            if tasks[i].cpu == tasks[h].cpu:
                tasks[i].hpp_set.append(h)
    # * get list of lower-priority tasks, lp
    for i in range(len(tasks)):
        for l in range(len(tasks[i+1:])):
            tasks[i].lp_set.append(l)
    
    return tasks

def print_seg_gpu(seg: List[GPUseg]):
    print(f"ges: {len(seg)}")
    for s in seg:
        print(f"{s.G_hi:.2f}, {s.Gm_hi:.2f}, {s.Ge_hi:.2f}, {s.Gd_hi:.2f} | ", end="")
    print()

def print_seg_cpu(seg: List[CPUseg]):
    print(f"ces: {len(seg)}")
    for s in seg:
        print(f"{s.C_hi:.2f} | ", end="")
    print()

def print_taskset(tasks: List[Task]):
    print("----------------------------------------------------")
    for task in tasks:
        print(colored(f"task id {task.id}: Default prio = {task.prio}, GPU prio = {task.prio_gpu}", attrs=['bold']))
        print(f"CPU = {task.cpu}, T = {task.T}, U = {task.U}, C_hi = {task.C_hi:.2f}, G_hi = {task.G_hi:.2f}, Gm_hi = {task.Gm_hi:.2f}, Ge_hi = {task.Ge_hi:.2f}, Gd_hi = {task.Gd_hi:.2f}")
        print_seg_cpu(task.ces)
        print_seg_gpu(task.ges)