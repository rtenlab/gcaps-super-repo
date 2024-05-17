import math
from sched_common import *

# the worst-case response time of tau_x's k-th ges
# Lakshmanan 2009
def mpcp_W_ges(tasks: List[Task], taux: Task, k: int) -> float:
    res = taux.ges[k].G_hi
    for h in range(len(tasks)): # find higher-prio tasks
        if tasks[h].prio <= taux.prio:
            continue
        max_G_hu = 0
        for u in range(len(tasks[h].ges)):
            if max_G_hu < tasks[h].ges[u].G_hi:
                max_G_hu = tasks[h].ges[u].G_hi
        res += max_G_hu
    return res

# the direct blocking on tau_i's j-th GPU segment
# it's independent from j, since there is only one resource to access
def mpcp_direct_blocking_segment(tasks: List[Task], taui: Task, j: int) -> float:
    max_W_lu = 0
    # compute Eq 5 first term
    for l in range(len(tasks)):
        if tasks[l].prio > taui.prio:
            continue
        for u in range(len(tasks[l].ges)):
            W_lu = mpcp_W_ges(tasks, tasks[l], u)
            if W_lu > max_W_lu:
                max_W_lu = W_lu

    # second term
    B_prev = 0
    B_curr = 0
    while True:
        B_curr = max_W_lu
        for h in taui.hp_set:
            for u in range(len(tasks[h].ges)):
                W_hu = mpcp_W_ges(tasks, tasks[h], u)
                B_curr += (math.ceil((B_prev + tasks[h].WR - tasks[h].C_hi - tasks[h].Gm_hi) / tasks[h].T) + 1) * W_hu
        if B_curr == B_prev:
            return B_curr
        B_prev = B_curr


def mpcp_direct_blocking(tasks: List[Task], taui: Task) -> float:
    res = 0
    for j in range(len(taui.ges)):
        res += mpcp_direct_blocking_segment(tasks, taui, j)
    return res


# blocking by the boosting of Gm
def mpcp_prioritized_blocking(tasks: List[Task], taui: Task) -> float:
    max_Gm_l = 0
    for l in range(len(tasks)):
        if tasks[l].prio > taui.prio or tasks[l].cpu != taui.cpu:
            continue
        max_Gm_lu = 0
        for u in range(len(tasks[l].ges)):
            if max_Gm_lu < tasks[l].ges[u].Gm_hi:
                max_Gm_lu = tasks[l].ges[u].Gm_hi
        max_Gm_l += max_Gm_lu

    return max_Gm_l * (len(taui.ges) + 1)


################################################################
# MPCP (RTAS 18) Hybrid 
################################################################
# WRi is the one being computed recursively
def mpcp_hybrid_alpha(tasks: List[Task], taui: Task, h: int, WRi):
    return math.ceil((WRi + tasks[h].WR - (tasks[h].C_hi + tasks[h].Gm_hi)) / tasks[h].T)

def mpcp_hybrid_B_dmh(tasks: List[Task], taui: Task, WRi: float) -> float: # fixme WRi
    B_dmh = 0
    for h in taui.hp_set:
        for k in range(len(tasks[h].ges)):
            sum_beta = 0
            for j in range(len(taui.ges)):
                sum_beta += math.ceil(mpcp_direct_blocking_segment(tasks, taui, j))
            alpha = mpcp_hybrid_alpha(tasks, taui, h, WRi)
            B_dmh += min(sum_beta, alpha) * mpcp_W_ges(tasks, tasks[h], k)
    return B_dmh

# compute L in Def. 3
# returns Q and X
def mpcp_hybrid_QX(tasks: List[Task], taui: Task) -> tuple[List, List]:
    # first get Q_i: a set of wcrt of gpu access that belongs tasks with lower priority than taui
    Q = []
    X = [] # stores the task index
    for l in range(len(tasks)):
        if tasks[l].prio >= taui.prio:
            continue
        for sid in range(len(tasks[l].ges)):
            Q.append(mpcp_W_ges(tasks, tasks[l], sid))
            X.append(l)

    sorted_X = [pair[1] for pair in sorted(zip(Q, X), key=lambda pair: pair[0], reverse=True)]
    Q.sort(reverse=True)
    return Q, sorted_X

def mpcp_hybrid_B_dml(tasks: List[Task], taui: Task, WRi: float) -> float:
    B_dml = 0
    Q, X = mpcp_hybrid_QX(tasks, taui) # L_{i,j,k} = QX[0][k]
    if Q == [] or X == []: # now lower-priority tasks of taui
        return 0
    psi_list = []
    for k in range(len(Q)):
        l = X[k] # = X_{i,j,k} in Eq. (12)
        # print(f" k = {k}, Q = {Q}, X = {X}")
        theta = math.ceil((WRi + tasks[l].T - (tasks[l].C_hi + tasks[l].Gm_hi)) / tasks[l].T)
        psi = max( min( len(taui.ges) - sum(psi_list), theta ), 0) # k-th not appended yet
        psi_list.append(psi)
        B_dml += psi * Q[k]

    return B_dml

def mpcp_hybrid_B_dm(tasks: List[Task], taui: Task, WRi: float) -> float:
    return mpcp_hybrid_B_dmh(tasks, taui, WRi) + mpcp_hybrid_B_dml(tasks, taui, WRi)

# S is the ordered set of cpu critical sections of taui
def mpcp_hybrid_Sk(tasks: List[Task], taui: Task, k: int) -> List[float]:
    sorted_ges = sorted(taui.ges, key=lambda g: g.Gm_hi, reverse=True)
    S = []
    for g in sorted_ges:
        S.append(g.Gm_hi)
    return S[k]


def mpcp_hybrid_B_pm(tasks: List[Task], taui: Task, WRi: float):
    B_pm = 0
    for l in range(len(tasks)):
        if tasks[l].prio > taui.prio or tasks[l].cpu != taui.cpu:
            continue
        phi_list = []
        for k in range(len(tasks[l].ges)):
            # compute phi
            theta = math.ceil((WRi + tasks[l].T - (tasks[l].C_hi + tasks[l].Gm_hi)) / tasks[l].T)
            phi = max( min( len(taui.ges) + 1 - sum(phi_list), theta ), 0)
            phi_list.append(phi)
            B_pm += phi * mpcp_hybrid_Sk(tasks, tasks[l], k)

    return B_pm

'''
The taskset must be ordered with priority
@param tasks: a list of sorted tasks with priority
@param rt_num: the number of real-time tasks in the task list
'''
def rt_test(tasks: List[Task], no_suspension: bool = False) -> bool:
    for i in range(len(tasks)):
        if tasks[i].prio == -1:
            continue
        
        B_l = 0
        B_r = 0
        # request-driven
        # B_l = mpcp_prioritized_blocking(tasks, tasks[i])
        # B_r = mpcp_direct_blocking(tasks, tasks[i])

        WR_prev = 0
        WR_curr = 0

        while True:
            WR_curr = tasks[i].C_hi + tasks[i].G_hi + B_l + B_r
            # compute the preemption time
            # note: only self-suspension for now
            for h in tasks[i].hpp_set:
                if no_suspension == False: # with suspension
                    WR_curr += math.ceil((WR_prev + (tasks[h].WR - tasks[h].C_hi - tasks[h].Gm_hi)) / tasks[h].T) * (tasks[h].C_hi + tasks[h].Gm_hi)
                else: # busy-waiting
                    WR_curr += math.ceil((WR_prev + (tasks[h].WR - tasks[h].C_hi - tasks[h].G_hi)) / tasks[h].T) * (tasks[h].C_hi + tasks[h].G_hi)

            WR_curr += mpcp_hybrid_B_pm(tasks, tasks[i], WR_prev) + mpcp_hybrid_B_dm(tasks, tasks[i], WR_prev)

            if WR_prev == WR_curr and WR_curr <= tasks[i].T:
                tasks[i].WR = WR_curr
                # print("WCRT of task {}: {}".format(tasks[i].id, tasks[i].WR, tasks[i].T))
                break
            elif WR_curr > tasks[i].T:
                tasks[i].WR = -1
                # print("MPCP task {} not schedulable, R ({}) > D ({})".format(tasks[i].id, WR_curr, tasks[i].T))
                return False # taskset not schedulable
            WR_prev = WR_curr

    return True

