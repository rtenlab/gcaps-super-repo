'''
This file is about schedulability tests for all the mentioned scheduling approaches in the paper.

Including:
1. gcaps_ioctl (busy and suspend)
3. default round-robin (busy and suspend)
4. mpcp (busy and suspend)
5. fmlp+ (busy and suspend)
'''

from dataclasses import dataclass
from enum import IntEnum
import numpy as np
import argparse

from sched_common import *

from gcaps import rt_test as rt_test_gcaps
from gcaps import rt_test_gpu_seg_prio as rt_test_gcaps_gprio
from mpcp import rt_test as rt_test_mpcp
from fmlp import rt_test as rt_test_fmlp
from default_driver import rt_test as rt_test_dft

from taskset_gen import taskset_generation, print_taskset, Params

'''
expr1 mainly compares the proposed approach against the prior work, including:
* gcaps_ioctl, busy and suspend
* mpcp (RTAS 18 hybrid), busy and suspend
* FMLP+, busy and suspend

# 0: number of best-effort tasks in the taskset
# 1: average number of tasks per core
# 2: ratio of lower bound to upper bound
# 3: ratio of cpu execution to gpu execution
# 4: bimodal distribution
# 5: number of cpus
'''

@dataclass
class policy_type(IntEnum):
    mpcp_suspend = 0
    mpcp_busy = 1
    fmlp_suspend = 2
    fmlp_busy = 3
    ioctl_suspend_improved = 5
    ioctl_busy_improved = 6

    ioctl_suspend_baseline = 7
    ioctl_busy_baseline = 8

    ioctl_suspend_gpu_prio = 9
    ioctl_busy_gpu_prio = 10

    tsg_rr_suspend = 12
    tsg_rr_busy = 13
    

def select_policy(tasks: List[Task], policy_id: int) -> int:
    ret = False
    if policy_id == policy_type.mpcp_suspend:
        ret = rt_test_mpcp(tasks, False)
    elif policy_id == policy_type.mpcp_busy:
        ret = rt_test_mpcp(tasks, True)
    elif policy_id == policy_type.fmlp_suspend:
        ret = rt_test_fmlp(tasks, False)
    elif policy_id == policy_type.fmlp_busy:
        ret = rt_test_fmlp(tasks, True)
    elif policy_id == policy_type.ioctl_suspend_baseline:
        ret = rt_test_gcaps(tasks, False, 0)
    elif policy_id == policy_type.ioctl_busy_baseline:
        ret = rt_test_gcaps(tasks, True, 0)
    elif policy_id == policy_type.ioctl_suspend_gpu_prio:
        ret = rt_test_gcaps(tasks, False, 0)
        if ret == False:
            ret = rt_test_gcaps_gprio(tasks, False, 0)
    elif policy_id == policy_type.ioctl_busy_gpu_prio:
        ret = rt_test_gcaps(tasks, True, 0)
        if ret == False:
            ret = rt_test_gcaps_gprio(tasks, True, 0)
    elif policy_id == policy_type.tsg_rr_suspend:
        ret = rt_test_dft(tasks, False)
    elif policy_id == policy_type.tsg_rr_busy:
        ret = rt_test_dft(tasks, True)

    tasks = clear_task_response_time(tasks)
    return ret

#####################################################################
# Experiment 1 and 2 are contained in the function
#
# Experiment 1: general schedulability test
# Experiment 3: Gain of separate GPU prio assignment
#####################################################################
def expr1(group_id: int, expr_id: int, loop_num: int):
    print("Running general experiments against prior works...")
    if expr_id == 1:
        print("expr_id: {} (number of best-effort tasks in the taskset)".format(expr_id))
    elif expr_id == 2:
        print("expr_id: {} (Number of tasks per cpu)".format(expr_id))
    elif expr_id == 3:
        print("expr_id: {} (ratio of gpu execution to cpu execution)".format(expr_id))
    elif expr_id == 4:
        print("expr_id: {} (utilization per cpu)".format(expr_id))
    elif expr_id == 5:
        print("expr_id: {} (number of cpus)".format(expr_id))
    elif expr_id == 6:
        print(f"expr_id: {expr_id} (ratio of gpu using tasks)")
    else:
        print("expr_id: {} unknown, exiting".format(expr_id))
        exit()

    if group_id == 1: # experiment 1: general schedulability test
        policies = [
            # policy_type.mpcp_busy,
            # policy_type.mpcp_suspend,
            # policy_type.fmlp_busy,
            # policy_type.fmlp_suspend,
            # policy_type.ioctl_busy_gpu_prio,
            # policy_type.ioctl_suspend_gpu_prio,
            # policy_type.tsg_rr_busy,
            # policy_type.tsg_rr_suspend
            0, 1, 2, 3, 9, 10, 12, 13
        ]
    elif group_id == 3:  # experiment 3: gain of GPU segment priority assignment
        policies = [
            # policy_type.ioctl_busy_baseline,
            # policy_type.ioctl_suspend_baseline,
            # policy_type.ioctl_busy_gpu_prio,
            # policy_type.ioctl_suspend_gpu_prio
            7, 8, 9, 10
        ]

    print(f"Policies to be tested: {policies}")

    expr_gid = 1
    params = Params()
    # results presentation 2d list: rows - config, col - policy
    results = [[0 for j_ in range(len(policies))] for i_ in range(20)]
    num_configs = 0

    if expr_id == 1: # percentage of be tasks
        # we have to use maximum number of tasks per core here
        params.range_num_tasks_per_cpu[0] = params.range_num_tasks_per_cpu[1]
        config_percentage_be_tasks = list(np.arange(0, 0.9, 0.1)) 
        for idx in range(len(config_percentage_be_tasks)):
            for niter in range(loop_num):                
                tasks = taskset_generation(params, expr_gid, -1, config_percentage_be_tasks[idx])
                for pid in range(len(policies)):
                    ret = select_policy(tasks, policies[pid])
                    if ret == True:
                        results[idx][pid] += 1
            print(f"# Setting {idx} completed...", end="")
        num_configs = len(config_percentage_be_tasks)
    elif expr_id == 2: # number of tasks in the taskset
        config_num_tasks = list(np.arange(2, 20 + 1, 2)) 
        for idx in range(len(config_num_tasks)):
            for niter in range(loop_num):
                tasks = taskset_generation(params, expr_gid, config_num_tasks[idx], 0)
                for pid in range(len(policies)):
                    ret = select_policy(tasks, policies[pid])
                    if ret == True:
                        results[idx][pid] += 1
            print(f"# Setting {idx} completed...", end="")
        num_configs = len(config_num_tasks)
    elif expr_id == 3: 
        config_ratio_g_to_c = list(np.arange(0, 1 + 0.1, 0.1))
        for idx in range(len(config_ratio_g_to_c)):
            params.range_ratio_g_to_c[0] = params.range_ratio_g_to_c[1] = config_ratio_g_to_c[idx]
            for niter in range(loop_num):
                tasks = taskset_generation(params, expr_gid, -1, 0)
                for pid in range(len(policies)):
                    ret = select_policy(tasks, policies[pid])
                    if ret == True:
                        results[idx][pid] += 1
            print(f"# Setting {idx} completed...", end="")
        num_configs = len(config_ratio_g_to_c)
    elif expr_id == 4: 
        config_util_per_cpu = list(np.arange(0.1, 1 + 0.1, 0.1))
        for idx in range(len(config_util_per_cpu)):
            params.range_util_per_cpu[0] = params.range_util_per_cpu[1] = config_util_per_cpu[idx]
            for niter in range(loop_num):
                tasks = taskset_generation(params, expr_gid, -1, 0)
                for pid in range(len(policies)):
                    ret = select_policy(tasks, policies[pid])
                    if ret == True:
                        results[idx][pid] += 1
            print(f"# Setting {idx} completed...", end="")
            num_configs = len(config_util_per_cpu)
    elif expr_id == 5: # number of cpus
        config_ncpu = list(np.arange(1, 9 + 1, 1))
        for idx in range(len(config_ncpu)):
            params.ncpu = config_ncpu[idx]
            for niter in range(loop_num):
                tasks = taskset_generation(params, expr_gid, -1, 0)
                for pid in range(len(policies)):
                    ret = select_policy(tasks, policies[pid])
                    if ret == True:
                        results[idx][pid] += 1
            print(f"# Setting {idx} completed...", end="")
        num_configs = len(config_ncpu)
    elif expr_id == 6: # ratio of gpu using tasks
        config_ratio_of_gpu_tasks = list(np.arange(0.1, 1 + 0.1, 0.1))
        for idx in range(len(config_ratio_of_gpu_tasks)):
            params.range_ratio_of_gpu_tasks[0] = params.range_ratio_of_gpu_tasks[1] = config_ratio_of_gpu_tasks[idx]
            for niter in range(loop_num):
                tasks = taskset_generation(params, expr_gid, -1, 0)
                for pid in range(len(policies)):
                    ret = select_policy(tasks, policies[pid])
                    if ret == True:
                        results[idx][pid] += 1
            print(f"# Setting {idx} completed...", end="")
        num_configs = len(config_ratio_of_gpu_tasks)

    # print each policy first as col
    print()
    for n_ in range(num_configs):
        for m_ in range(len(results[0])):
            val = round(results[n_][m_] / loop_num * 100, 2)
            if m_ != len(results[0]) - 1:
                print(f"{val}, ", end='')
            else:
                print(f"{val}")
        # print(*results[n_] / loop_num * 100, sep=",") # Print with comma separator & without brackets

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RT Test Simulation for GCAPS')
    parser.add_argument('-g', '--group', type=int, help='Experiment group id', required=True)
    parser.add_argument('-e', '--expr', type=int, help='Experiment id', required=True)
    parser.add_argument('-n', '--ntasks', type=int, help='Number of tasksets', required=True)
    args = parser.parse_args()
    if args.group == 1 or args.group == 3:
        expr1(args.group, args.expr, args.ntasks)
            
        
    