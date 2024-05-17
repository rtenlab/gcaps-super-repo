
"""  
This file includes the RTA for default round-robin scheduling approach in GPU driver under busy-waiting and self-suspension mode
"""

import math
from sched_common import *

# max length of TSG (ms)
L = 1
# TSG ctx sw overhead (ms)
theta = 0.2


######################################################################
# GCAPS Eq. 3 
# @param `G` exec. time of a segment, NOT a task
# @param `nu` number of GPU-using tasks in the system
######################################################################
def compute_I(nu: int, G: float) -> float:
    return (L + theta) * nu * math.ceil(G / L)


def rt_test(tasks: List[Task], no_suspension: bool = False) -> bool:

    for i in range(len(tasks)):
        # do not need to bound RT for BE tasks
        if tasks[i].prio == -1:
            continue

        # number of GPU-using tasks except task i
        Nie = 0 
        for k in range(len(tasks)):
            if len(tasks[k].ges) > 0 and k != i:
                Nie += 1
        
        WR_prev = 0
        WR_curr = 0

        # bounding interleaved exection
        Iie = 0
        for gs in tasks[i].ges:
            Iie += compute_I(Nie, gs.Ge_hi)

        while True:
            WR_curr = tasks[i].C_hi + tasks[i].G_hi + Iie

            for h in tasks[i].hpp_set:
                # bounding indirect preemption and blocking, only for busy-waiting
                if no_suspension == True:
                    if len(tasks[h].ges) == 0:
                        continue
                    # ECRTS camera-ready version; merged indirect delay
                    Nid = 0
                    for hh in tasks[i].hp_set:
                        if hh not in tasks[i].hpp_set and len(tasks[hh].ges) > 0:
                            Nid += 1
                    for l in tasks[i].lp_set:
                        if len(tasks[l].ges) > 0:
                            Nid += 1
                    Iid = 0
                    for gs in tasks[h].ges:
                        Iid += compute_I(Nid, gs.Ge_hi)
                    
                    WR_curr += math.ceil((WR_prev)/tasks[h].T) * (Iid + tasks[h].G_hi)
                else: # this item does not exist under self-suspension mode
                    pass

                # bounding the CPU preemption
                if no_suspension == True: # busy-waiting
                    WR_curr += math.ceil((WR_prev)/tasks[h].T) * (tasks[h].C_hi + tasks[h].Gm_hi)
                else: # self-suspending
                    J = tasks[h].T - (tasks[h].C_hi + tasks[h].Gm_hi)
                    WR_curr += math.ceil((WR_prev + J)/tasks[h].T) * (tasks[h].C_hi + tasks[h].Gm_hi)

            if WR_prev == WR_curr and WR_curr <= tasks[i].T:
                tasks[i].WR = WR_curr
                break
            elif WR_curr > tasks[i].T:
                tasks[i].WR = -1
                return False # taskset not schedulable
            WR_prev = WR_curr
            
    return True