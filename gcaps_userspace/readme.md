# GCAPS Userspace Implementation
This folder includes the userspace implementation for GCAPS method [1].
The two macros to mark the GPU segment boundaries are defined [here](common/include/support.h#L99-L131).
:exclamation::exclamation: **Before proceeding, please finish deploying the driver code first.**

## Preparations
On target Jetson device:
```bash
sudo nvpmodel -m 2 # set power mode
sudo /usr/bin/jetson_clocks # set clock speed to the max
sudo sysctl -w kernel.sched_rt_runtime_us=-1 # enable real-time priorities
```

## How to Run
First compile the executable:
```bash
cd 
git clone https://github.com/rtenlab/gcaps-super-repo.git
cd gcaps-super-repo/gcaps_userspace
make main
```

Then the case study can be run with the following command:
```bash
./main -f taskset.csv -d <duration> -i <ioctl enabled> -s <suspension enabled> -b <synchronization-based>
```

The details regarding each input argument are as follows:
- [-d]: running duration of the case study in seconds. We used 30 for the evaluation in the paper.
- [-i]: whether the GCAPS IOCTL-based approach is enabled. 1 - enabled, 0 - not enabled.
- [-s]: whether self-suspension mode is enabled. 1 - enabled, 0 - not enabled.
- [-b]: whether synchronization-based mode is enabled (the approach in previous literature [2]). The program will be aborted if [-b] and [-i] are set at the same time.

### Example Usage
```bash
# Run gcaps ioctl-based approach with self-suspension for 10 seconds
./main -f taskset.csv -d 10 -i 1 -s 1 -b 0

# Run default tsg round-robin approach with busy-waiting for 10 seconds
./main -f taskset.csv -d 10 -i 0 -s 0 -b 0
```

### Interpreting the Results
```bash
$ ./main -f taskset.csv -d 10 -i 1 -s 1 -b 0
Program configurations:
taskset: taskset.csv
duration: 10
ioctl enabled: 1
suspension: 1
sync mode: 0
---------------------------------------
...
[3525:4], 38.755001, 62.129757, 65.802002, 88.791000
[3526:5], 65.133003, 77.673798, 67.499397, 90.161003
[3522:2], 12.165000, 15.928424, 17.643499, 20.466000
[3523:3], 63.625999, 64.924858, 64.586853, 67.499001
[3524:6], 51.171001, 59.056057, 57.769749, 64.033997
[3521:1], 7.837000, 8.040460, 7.978450, 9.189000
```
For each row, it shows: [pid, task id], min, mean, perc95, max. The unit is in milliseconds.


## References
[1] Yidi Wang, Cong Liu, Daniel Wong, and Hyoseung Kim. GCAPS: GPU Context-Aware Preemptive Priority-based Scheduling for Real-Time Tasks. In Euromicro Conference on Real-Time Systems (ECRTS), 2024.
[2] Björn B Brandenburg. The FMLP+: An asymptotically optimal real-time locking protocol for suspension-aware analysis. In 2014 26th Euromicro Conference on Real-Time Systems, pages 61–71. IEEE, 2014.
