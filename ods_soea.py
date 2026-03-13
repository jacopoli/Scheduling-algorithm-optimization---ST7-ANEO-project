class Task:
    def __init__(self, id, execution_times, energy_costs):
        self.id = id
        self.execution_times = execution_times              # {proc_id: exec_time_at_fmax}
        self.energy_costs = energy_costs                    # {proc_id: energy_at_fmax}
        self.successors = []                                # List of Task objects
        self.predecessors = []                              # List of Task objects
        self.comm_costs = {}                                # {succ_id: communication_cost}
        self.urv = 0
        self.out_degree = 0

    def add_predecessor(self, predecessor: "Task"):
        self.predecessors.append(predecessor)

    def add_successor(self, successor: "Task"):
        self.successors.append(successor)

    def __str__(self):
        pred_ids = [p.id for p in self.predecessors]
        succ_ids = [s.id for s in self.successors]
        return (f"Task(id={self.id}, "
                f"predecessors={pred_ids}, "
                f"successors={succ_ids}, "
                f"execution_times={self.execution_times}, "
                f"energy_costs={self.energy_costs})")


def calculate_urv(tasks):
    """
    Computes the Up-Rank Value for each task (Equation 29).
    Assumption: tasks is ordered such that task[i] only depends on task[j] with j < i,
    so reversed(tasks) guarantees that successors are processed before their predecessors.
    """
    for task in reversed(tasks):
        avg_exec = sum(task.execution_times.values()) / len(task.execution_times)
        max_succ_val = 0
        for succ in task.successors:
            val = task.comm_costs.get(succ.id, 0) + succ.urv
            if val > max_succ_val:
                max_succ_val = val
        task.urv = avg_exec + max_succ_val


def get_ods_scheduling(tasks, processors, theta=0, l_idx=0):
    """
    Implementation of the ODS algorithm (Algorithm 2 from the paper).

    - tasks     : list of Task objects, ordered such that task[i] only depends on task[j] with j < i
    - processors: list of processor ids
    - theta     : reliability weighting factor (θ >= 0). If theta=0, ODS is equivalent to HEFT.
    - l_idx     : partitioning threshold (0 <= l <= n-1).
                  Tasks in ODQ[0..l_idx] are assigned with time priority (Equation 30),
                  the rest are assigned with energy priority (Equation 31).
    """

    # 1. Compute structural properties
    for t in tasks:
        t.out_degree = len(t.successors)

    calculate_urv(tasks)

    # 2. Priority queues
    # ODQ: sorted by out_degree descending, ties broken by urv descending (paper Section V)
    odq = sorted(tasks, key=lambda x: (x.out_degree, x.urv), reverse=True)
    # RQ: sorted by urv descending (allocation order)
    rq = sorted(tasks, key=lambda x: x.urv, reverse=True)

    # High time-priority group: ODQ[0..l_idx] inclusive
    high_degree_ids = {t.id for t in odq[:l_idx + 1]}

    # Tracking structures
    task_finish_times = {}                      # {task_id: finish_time}
    proc_available_time = {p: 0 for p in processors}
    allocation = {}                             # {task_id: proc_id}

    def get_est(t, p):
        """
        Earliest Start Time of task t on processor p.
        Formula: max{ ft_j + w_ji | τj ∈ pre(τi) }, with w_ji=0 if same processor.
        """
        ready_time = 0
        for pred in t.predecessors:
            # Communication cost is zero if both tasks are on the same processor
            comm_delay = pred.comm_costs.get(t.id, 0) if allocation[pred.id] != p else 0
            ready_time = max(ready_time, task_finish_times[pred.id] + comm_delay)
        # Processor p must also be available
        return max(ready_time, proc_available_time[p])

    # 3. Allocation (main loop - Algorithm 2)
    for task in rq:
        best_proc = None
        earliest_finish_time = float('inf')
        min_energy = float('inf')

        if task.id in high_degree_ids:
            # Equation (30): min{ ft_τi,k + θ * (1 - R_τi,k) * T_τi,k }
            for p in processors:
                ft = get_est(task, p) + task.execution_times[p]
                reliability = getattr(task, 'reliability', {}).get(p, 1.0)
                score = ft + theta * (1 - reliability) * task.execution_times[p]
                if score < earliest_finish_time:
                    earliest_finish_time = score
                    best_proc = p
        else:
            # Equation (31): min{ E_τi,k }
            for p in processors:
                if task.energy_costs[p] < min_energy:
                    min_energy = task.energy_costs[p]
                    best_proc = p
            earliest_finish_time = get_est(task, best_proc) + task.execution_times[best_proc]

        allocation[task.id] = best_proc
        task_finish_times[task.id] = earliest_finish_time
        proc_available_time[best_proc] = earliest_finish_time

    return allocation, task_finish_times