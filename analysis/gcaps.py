"""
This file only contains the regular RTA for gcaps with busy-waiting or self-suspension
"""

from typing import List
import math

from sched_common import *  

import decimal
# Set the default precision for decimal rounding to 2 decimal places
decimal.getcontext().prec = 2


#################################################################################
# Regular gcaps RTA

# @param `tasks`: a list of tasks sorted by the default priority
# @param `mode`: "suspend", "busy"
# @param `approach`: gcaps_ioctl (0)
# @return: True if the taskset is schedulable, False if not
#################################################################################
def rt_test(tasks: List[Task], no_suspension: bool = False, approach: int = 0) -> bool:
    if approach != 0:
        print("Invalid approach id! Exiting...")
        exit()

    for i in range(len(tasks)):
        if tasks[i].prio == -1:
            continue

        WR_prev = 0
        WR_curr = 0

        while True:
            WR_curr = tasks[i].C_hi + tasks[i].G_hi
            if approach == 0: # ioctl approach
                WR_curr += epsilon + 3 * epsilon * len(tasks[i].ges)
            # compute the preemption time by cpu execution
            for h in tasks[i].hpp_set:
                ''' basic analysis equations below '''
                if approach == 0 and no_suspension == False: # ioctl-suspend
                    if len(tasks[h].ges) == 0: # eta^g_h == 0, cpu-only task
                        WR_curr += math.ceil((WR_prev) / tasks[h].T) * (tasks[h].C_hi)
                    else: # gpu-using task
                        J = (tasks[h].WR - tasks[h].C_hi - tasks[h].Gm_hi)
                        WR_curr += math.ceil((WR_prev + J) / tasks[h].T) * (tasks[h].C_hi + tasks[h].Gm_hi + 2 * epsilon * len(tasks[h].ges))
                    if len(tasks[h].ges) > 0 and len(tasks[i].ges) > 0:
                        J = tasks[h].WR - tasks[h].Ge_hi
                        WR_curr += math.ceil((WR_prev + J) / tasks[h].T) * (tasks[h].Ge_hi)
                elif approach == 0 and no_suspension == True: # ioctl-busy
                    if len(tasks[h].ges) == 0: # eta^g_h == 0, cpu-only task
                        WR_curr += math.ceil((WR_prev) / tasks[h].T) * (tasks[h].C_hi)
                    else:
                        WR_curr += math.ceil((WR_prev) / tasks[h].T) * (tasks[h].C_hi + tasks[h].G_hi + 2 * epsilon * len(tasks[h].ges))

            # compute the preemption time by gpu execution
            for h in tasks[i].hp_set:
                if tasks[h].G_hi == 0: # applies for all 3 analysis
                    continue
                
                ''' basic analysis equations below '''
                # no gpu preemption for tasks that do not have gpu segments
                if approach == 0 and no_suspension == False and tasks[i].G_hi > 0: # ioctl-suspend
                    if h in tasks[i].hpp_set:   
                        continue # next h
                    J = tasks[h].WR - tasks[h].Ge_hi
                    WR_curr += math.ceil((WR_prev + J) / tasks[h].T) * (tasks[h].Ge_hi + 2 * epsilon * len(tasks[h].ges))
                elif approach == 0 and no_suspension == True: # ioctl-busy
                    if h in tasks[i].hpp_set:   
                        continue # next h
                    J = tasks[h].WR - tasks[h].Ge_hi
                    WR_curr += math.ceil((WR_prev + J) / tasks[h].T) * (tasks[h].Ge_hi + 2 * epsilon * len(tasks[h].ges))

            if WR_prev == WR_curr and WR_curr <= tasks[i].T:
                tasks[i].WR = WR_curr
                break
            elif WR_curr > tasks[i].T:
                tasks[i].WR = -1
                return False # taskset not schedulable
            WR_prev = WR_curr

    return True



######################################################################
# Analysis of GPU Prio Assignation
######################################################################

######################################################################
# A recursive function to check whether task idx can take the lowest priority in taskset `tasks`
# by using Audsley's approach (2007)

# @param `tasks_`: list that keeps all the real-time tasks with higher-priority on gpu than tasks_[idx]
# @param `idx`: the idx of the task in `tasks_` which is being checked whether can take the lowest priority
######################################################################
def can_take_lowest_prio(tasks: List[Task], tasks_: List[Task], ordering: List[Task], idx: int, prio_to_be_taken: int, no_suspension: bool = False, approach: int = 0) -> bool:
    if len(tasks_) == 0: # all the real-time tasks have been assigned priorities
        ret = inversion_detection(ordering)
        return ret
    
    take = False

    WR_prev = 0
    WR_curr = 0
    while True:
        gh_set = [] # set of tasks with higher gpu priority, storing the index in tasks
        for gh in range(len(tasks_)):
            if gh == idx:
                continue
            gh_set.append(tasks_[gh].id) # id = index in list tasks
        # note that we replaced R_h with D_h since R_h is agnostic 
        WR_curr = tasks_[idx].C_hi + tasks_[idx].G_hi
        if approach == 0: # ioctl approach
            WR_curr += epsilon + 3 * epsilon * len(tasks_[idx].ges)
        # since `tasks_` is sorted with RM order, higher-priority tasks have index `h < idx`. CPU preemption occurs on the same core
        for h in tasks_[idx].hpp_set: # h != idx # no kthread approach. This h is the index in list tasks
            ''' basic analysis equations'''
            if approach == 0 and no_suspension == False: # ioctl-suspend
                if len(tasks[h].ges) == 0: # eta^g_h == 0, cpu-only task
                    WR_curr += math.ceil((WR_prev) / tasks[h].T) * (tasks[h].C_hi)
                else: # gpu-using task
                    J = (tasks[h].T - tasks[h].C_hi - tasks[h].Gm_hi)
                    WR_curr += math.ceil((WR_prev + J) / tasks[h].T) * (tasks[h].C_hi + tasks[h].Gm_hi + 2 * epsilon * len(tasks[h].ges))
                if len(tasks[h].ges) > 0 and len(tasks_[idx].ges) > 0:
                    J = tasks[h].WR - tasks[h].Ge_hi
                    WR_curr += math.ceil((WR_prev + J) / tasks[h].T) * (tasks[h].Ge_hi)
            elif approach == 0 and no_suspension == True: # ioctl-busy
                if len(tasks[h].ges) == 0: # eta^g_h == 0, cpu-only task
                    WR_curr += math.ceil((WR_prev) / tasks[h].T) * (tasks[h].C_hi)
                else:
                    WR_curr += math.ceil((WR_prev) / tasks[h].T) * (tasks[h].C_hi + tasks[h].G_hi + 2 * epsilon * len(tasks[h].ges))
                
        # since `tasks_[idx]` is not necessarily to be the last one in the list, and lower-priority tasks are already popped, any task in `tasks_` can have a higher gpu priority than `tasks_[idx]`
        for h in gh_set:
            if tasks[h].id == tasks_[idx].id: # excludes the task being checked
                continue
            if tasks[h].G_hi == 0:
                continue
            ''' basic analysis equations'''
            if approach == 0 and no_suspension == False and tasks_[idx].G_hi > 0: # ioctl-suspend
                if h in tasks_[idx].hpp_set:   
                    continue
                J = tasks[h].T - tasks[h].Ge_hi
                WR_curr += math.ceil((WR_prev + J) / tasks[h].T) * (tasks[h].Ge_hi+ 2 * epsilon * len(tasks[h].ges))
            elif approach == 0 and no_suspension == True: # ioctl-busy
                if h in tasks_[idx].hpp_set:   
                    continue
                J = tasks[h].T - tasks[h].Ge_hi
                WR_curr += math.ceil((WR_prev + J) / tasks[h].T) * (tasks[h].Ge_hi+ 2 * epsilon * len(tasks[h].ges))

        if WR_prev == WR_curr and WR_curr <= tasks_[idx].T:
            # tasks_[idx].WR = WR_curr
            take = True
            break
        elif WR_curr > tasks_[idx].T:
            # tasks_[idx].WR = -1
            take = False
            break
        WR_prev = WR_curr

    # if it can, pop task idx from the set, and check for prio k-1; if not, check for the next task idx+1
    if take == True:
        # setting the gpu prio in orig task list for recording prupose
        for ti in range(len(tasks)):
            if tasks_[idx].id == tasks[ti].id:
                tasks[ti].prio_gpu = prio_to_be_taken
        ordering.append(tasks_.pop(idx))
        idx = len(tasks_) - 1
        prio_to_be_taken += 1
        return can_take_lowest_prio(tasks, tasks_, ordering, idx, prio_to_be_taken, no_suspension, approach)
    else:
        idx -= 1
        if idx < 0:
            return False # all the tasks_ were checked. this taskset is not schedulable
        return can_take_lowest_prio(tasks, tasks_, ordering, idx, prio_to_be_taken, no_suspension, approach)


##########################################################################
# gcaps RTA for separate GPU segment priority assignment

# This function should be triggered if `preemptive_gpu.rt_test()` returned False
# Input taskset is suppoed to be reset before sent as input argument
# @param `approach` - 0: ioctl, 1: kthread

# @return: True if the taskset is schedulable under new GPU priority assignation; False if not
##########################################################################
def rt_test_gpu_seg_prio(tasks: List[Task], no_suspension: bool = False, approach: bool = 0) -> int:
    if approach != 0:
        print("Invalid approach id! Exiting...")
        exit()
    # tasks in `tasks` are ordered by RM priority by default
    # create a list of tasks that only contains real-time tasks
    tasks_ = []
    for i in range(len(tasks)):
        if tasks[i].prio == -1:
            continue
        tasks_.append(tasks[i])
    ordering = []
    # Use the recursion to check whether it can be assigned with the lowest priority
    # starting from the task with lowest RM priority
    prio_to_be_taken = 1
    return can_take_lowest_prio(tasks, tasks_, ordering, len(tasks_) - 1, prio_to_be_taken, no_suspension, approach)


#######################################################################################
# Detect ordering inversion occuring on the same CPU core
# Such case can cause deadlock potentially

# This function detects whether there is any inversion of the ordering of tasks assigned to the same CPU
# e.g. task1 and task2 are assigned to the same CPU; CPU prio task1 > task2; GPU prio task1 < task2
# In this case, the taskset should be considered as unfeasible
#######################################################################################
def inversion_detection(ordering: List[Task]) -> int:
    # ordering stores the tasks in the REVERSE ordering of GPU prio
    cpu_last_taskid = {}

    for task in ordering:
        # If the CPU for this task has been seen before and the current task's id is smaller than the last seen id, return True
        if task.cpu in cpu_last_taskid and task.id > cpu_last_taskid[task.cpu]:
            return False # the inversion exists, unfeasible
        cpu_last_taskid[task.cpu] = task.id

    # If no such condition is found, the taskset is feasible
    return True
