import math

class Processor:
    def __init__(self, id, lambda_f, d, f_min, f_max, alpha, c, p_static):
        self.id = id
        self.lambda_f = lambda_f    # Average fault rate at fmax
        self.d = d                  # Hardware-related constant
        self.f_min = f_min          # Minimum available frequency
        self.f_max = f_max          # Maximum available frequency (normalized to 1.0)
        self.alpha = alpha          # Power exponent (~3 for real chips)
        self.c = c                  # Loading capacitance
        self.p_static = p_static    # Static + frequency-independent power consumption

    def lambda_at(self, f):
        """ Fault rate at frequency f (Equation 6) """
        return self.lambda_f * (10 ** (self.d * (1 - f) / (1 - self.f_min)))

    def reliability(self, f, exec_time_at_fmax):
        """ Reliability of a task at frequency f (Equation 7) """
        t = exec_time_at_fmax / f
        return math.exp(-self.lambda_at(f) * t)

    def energy(self, f, exec_time_at_fmax):
        """ Energy consumption at frequency f (Equation 4) """
        t = exec_time_at_fmax / f
        return (self.p_static + self.c * (f ** self.alpha)) * t

    def __str__(self):
        return (f"Processor(id={self.id}, lambda_f={self.lambda_f}, d={self.d}, "
                f"f_min={self.f_min}, f_max={self.f_max}, alpha={self.alpha}, "
                f"c={self.c}, p_static={self.p_static})")


def soea(tasks, allocation, proc_map, r_req, epsilon=1e-5):
    """
    Search-based Optimal Energy Allocation (Algorithm 1 from the paper).
    Given the allocation of tasks to processors, finds the optimal frequency
    for each task that minimizes energy while satisfying r_req.
    """

    task_map = {t.id: t for t in tasks}

    def o_at_f(f, proc):
        """ Lagrange multiplier o as a function of f (Equation 23) """
        lam = proc.lambda_at(f)
        numerator = proc.c * (proc.alpha - 1) * (f ** (proc.alpha - 1)) - proc.p_static / f
        denominator = r_req * lam * (proc.d * math.log(10) / (1 - proc.f_min) + 1 / f)
        return numerator / denominator

    def f_from_o(o, proc, task_id):
        """ Binary search to find f given o for a specific task/processor (inner loop) """
        f_lb = proc.f_min
        f_ub = proc.f_max
        while (f_ub - f_lb) > epsilon:
            f_mid = (f_lb + f_ub) / 2
            if o_at_f(f_mid, proc) < o:
                f_lb = f_mid
            else:
                f_ub = f_mid
        return (f_lb + f_ub) / 2

    def compute_reliability(freq_map):
        """ Total DAG reliability given a frequency map {task_id: f} """
        r = 1.0
        for task_id, f in freq_map.items():
            proc = proc_map[allocation[task_id]]
            T = task_map[task_id].execution_times[proc.id]
            r *= proc.reliability(f, T)
        return r

    # Search bounds for o (Equation 27)
    o_lb = min(o_at_f(proc.f_min, proc)
               for t in tasks for proc in [proc_map[allocation[t.id]]])
    o_ub = max(o_at_f(proc.f_max, proc)
               for t in tasks for proc in [proc_map[allocation[t.id]]])

    # Outer binary search over o
    while (o_ub - o_lb) > epsilon:
        o_mid = (o_lb + o_ub) / 2

        # For each task, find the frequency corresponding to o_mid
        freq_map = {}
        for task in tasks:
            proc = proc_map[allocation[task.id]]
            freq_map[task.id] = f_from_o(o_mid, proc, task.id)

        # Check if reliability constraint is satisfied
        if compute_reliability(freq_map) < r_req:
            o_lb = o_mid
        else:
            o_ub = o_mid

    # Final frequencies
    freq_map = {}
    for task in tasks:
        proc = proc_map[allocation[task.id]]
        freq_map[task.id] = f_from_o((o_lb + o_ub) / 2, proc, task.id)

    return freq_map