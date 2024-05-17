# Response Time Analysis
This folder includes the necessary files to run the schedulability analysis in GCAPS paper. The implemented approaches include:
- GCAPS: suspend + busy (proposed in our work [1])
- Default GPU round-robin: suspend + busy (first attempt to bound default GPU scheduling approach [1])
- FMLP+: suspend + busy [2]
- MPCP: suspend + busy [3]

## How to run
The experiments can be run with the following command:
```bash
cd analysis
python3 experiments.py -g <figure id> -e <experiment setting id> -n <number of taskset>
```

**Input argument [-g]** `figure id`:
- 1: General schedulability test (figure 8)
- 3: Gain of separate GPU priority assignment (figure 9)

**Input argument [-e]** `experiment id`:
- 1: Number of best-effort tasks
- 2: Number of tasks per CPU
- 3: Ratio of GPU exec. to CPU exec.
- 4: Util. per CPU
- 5: Number of CPUs
- 6: Ratio of GPU-using tasks

**Input argument [-n]** `number of taskset`:
We use 1000 for the experiments in the paper.

### Example Usage
```bash
$ python3 experiments.py -g 1 -e 1 -n 100
Running general experiments against prior works...
expr_id: 1 (number of best-effort tasks in the taskset)
Policies to be tested: [0, 1, 2, 3, 9, 10, 12, 13]
# Setting 0 completed...# Setting 1 completed...# Setting 2 completed...# Setting 3 completed...# Setting 4 completed...# Setting 5 completed...# Setting 6 completed...# Setting 7 completed...# Setting 8 completed...
0.0, 0.0, 6.0, 5.0, 5.0, 8.0, 2.0, 1.0
5.0, 5.0, 10.0, 10.0, 14.0, 16.0, 5.0, 9.0
2.0, 2.0, 7.0, 7.0, 16.0, 17.0, 3.0, 3.0
3.0, 3.0, 8.0, 8.0, 39.0, 43.0, 5.0, 5.0
12.0, 12.0, 15.0, 15.0, 57.0, 58.0, 16.0, 17.0
16.0, 16.0, 19.0, 18.0, 82.0, 86.0, 22.0, 16.0
17.0, 17.0, 21.0, 21.0, 85.0, 84.0, 30.0, 25.0
30.0, 29.0, 26.0, 25.0, 97.0, 100.0, 35.0, 33.0
56.0, 56.0, 37.0, 37.0, 99.0, 100.0, 62.0, 59.0
```
Every column stands for the schedulability under one scheduling approach, and corresponding settings can be found at `experiments.py` line 41.

## References
[1] Yidi Wang, Cong Liu, Daniel Wong, and Hyoseung Kim. GCAPS: GPU Context-Aware Preemptive Priority-based Scheduling for Real-Time Tasks. In Euromicro Conference on Real-Time Systems (ECRTS), 2024.
[2] Björn B Brandenburg. The FMLP+: An asymptotically optimal real-time locking protocol for suspension-aware analysis. In 2014 26th Euromicro Conference on Real-Time Systems, pages 61–71. IEEE, 2014.
[3] Pratyush Patel, Iljoo Baek, Hyoseung Kim, and Ragunathan Rajkumar. Analytical enhancements and practical insights for MPCP with self-suspensions. In IEEE Real-Time and Embedded Technology and Applications Symposium (RTAS), 2018.