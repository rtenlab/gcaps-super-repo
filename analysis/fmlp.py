import math

from sched_common import *

# the index of rows is cpu id
def get_pcpus(tasks: list[Task]) -> List[List[Task]]:
    # create lists: tasks on each cpu
    myset = set()
    for x in range(len(tasks)):
        myset.add(tasks[x].cpu)
    ncpu = len(myset)

    pcpus = [[] * 0 for _ in range(ncpu)]
    for x in range(len(tasks)):
        pcpus[tasks[x].cpu].append(tasks[x])
    
    return pcpus

def fmlp_tif(taui: Task, t: float, resp_i: float) -> List[float]:
    p_i = taui.T
    L_iqv = [] # to be returned
    # - L_iq: Maximum length of any request for resource q by t_i
	# - Find L_iq and append it to L_iqv max_cnt times
    L_iq = 0
    N_iq = 0
    for x in range(len(taui.ges)):
        N_iq += 1
        if taui.ges[x].G_hi > L_iq:
            L_iq = taui.ges[x].G_hi
    max_cnt = N_iq * math.ceil((t + resp_i) / p_i)
    for v in range(max_cnt):
        L_iqv.append(L_iq)
    return L_iqv
        
def fmlp_top(l: int, s: List[float]) -> List[float]:
    output = []
    s.sort(reverse=True)
    cnt = len(s) if (l > len(s)) else l
    
    for i in range(cnt):
        output.append(s[i])
    
    return output

def fmlp_total(l: int, s: List[float]) -> float:
    ret = 0
    s.sort(reverse=True)
    cnt = len(s) if (l > len(s)) else l
    
    for i in range(cnt):
        ret += s[i]
    return ret

def fmlp_tifs(tset: List[Task], t: float, l: int) -> List[float]:
    output = []
    for x in range(len(tset)):
        tif_output = fmlp_tif(tset[x], t, tset[x].T)
        top_output = fmlp_top(l, tif_output)
        output.extend(top_output)
    return output

def fmlp_db(tset_j: List[Task], taui: Task, r_i: float) -> int:
    N_iq = len(taui.ges)

    tifs_output = fmlp_tifs(tset_j, r_i, N_iq)
    return len(tifs_output)
        
def fmlp_D(taui: Task) -> int:
    return len(taui.ges)

def fmlp_B_r(tasks: List[Task], cpu_id: int, taui: Task, W: float) -> float:
    ret = 0
    pcpus = get_pcpus(tasks)

    for j in range(len(pcpus)):
        if cpu_id == j:
            continue
        for x in range(len(pcpus[j])):
            taux = pcpus[j][x]
            tif_result = fmlp_tif(taux, W, taux.T)
            ret += fmlp_total(min(fmlp_db(pcpus[j], taui, W), fmlp_D(taui)), tif_result)

    return ret

def fmlp_z_iq(tasks: List[Task], cpu_id: int, taui: Task, resp_i: float) -> int:
    N_iq = len(taui.ges)

    pcpus = get_pcpus(tasks)
    tset = []
    for j in range(len(pcpus)):
        if cpu_id == j:
            continue
        tset.extend(pcpus[j])
    
    tifs_output = fmlp_tifs(tset, resp_i, N_iq)
    return min(N_iq, len(tifs_output))

def fmlp_B_l(tasks: List[Task], taui: Task, W: float) -> float:
    ret = 0
    pcpus = get_pcpus(tasks)
    for x in range(len(tasks)): # lower priority tasks
        if tasks[x].cpu != taui.cpu:
            continue
        if tasks[x].prio >= taui.prio:
            continue
        tif_result = fmlp_tif(tasks[x], W, tasks[x].T)
        ret += fmlp_total(1 + fmlp_z_iq(tasks, taui.cpu, taui, W), tif_result)
    return ret



def rt_test(tasks: List[Task], no_suspension: bool = False) -> bool:
    for i in range(len(tasks)):
        if tasks[i].prio == -1:
            continue
        WR_prev = 0
        WR_curr = 0
        h = 0
        u = 0
        while True:
            B_l = fmlp_B_l(tasks, tasks[i], WR_prev)
            B_r = fmlp_B_r(tasks, tasks[i].cpu, tasks[i], WR_prev)
            WR_curr = tasks[i].C_hi + tasks[i].G_hi + B_l + B_r

            for h in tasks[i].hpp_set:
                if no_suspension == False:
                    WR_curr += math.ceil((WR_prev + (tasks[h].WR - tasks[h].C_hi - tasks[h].Gm_hi)) / tasks[h].T) * (tasks[h].C_hi + tasks[h].Gm_hi)
                else: # busy-waiting
                    WR_curr += math.ceil((WR_prev + (tasks[h].WR - tasks[h].C_hi - tasks[h].G_hi)) / tasks[h].T) * (tasks[h].C_hi + tasks[h].G_hi)
            
            if WR_prev == WR_curr and WR_curr <= tasks[i].T:
                tasks[i].WR = WR_curr
                # print(f"WCRT of task {tasks[i].id}: {tasks[i].WR}")
                break
            elif WR_curr > tasks[i].T:
                tasks[i].WR = -1
                # print("task {} failed at prio {}, R ({}) > D ({})".format(tasks[i].id, i, WR_curr, tasks[i].T))
                return False # taskset not schedulable
            WR_prev = WR_curr

    return True
        

