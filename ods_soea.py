import math

class Task:
    def __init__(self, id, execution_times, energy_costs):
        self.id = id
        self.execution_times = execution_times  # {proc_id: temps_a_fmax}
        self.energy_costs = energy_costs        # {proc_id: energie_a_fmax}
        self.successors = []    # Liste d'objets Task
        self.predecessors = []  # Liste d'objets Task
        self.comm_costs = {}    # {succ_id: cout_communication}
        self.urv = 0
        self.out_degree = 0

def calculate_urv(tasks):
    """ Calcule l'Up-Rank Value de façon récursive (Formule 29) """
    for task in reversed(tasks): # On part souvent des feuilles vers la racine
        avg_exec = sum(task.execution_times.values()) / len(task.execution_times)
        max_succ_val = 0
        for succ in task.successors:
            val = task.comm_costs[succ.id] + succ.urv
            if val > max_succ_val:
                max_succ_val = val
        task.urv = avg_exec + max_succ_val

def get_ods_scheduling(tasks, processors, theta=0, l_idx=0):
    # 1. Calculer les propriétés structurelles
    for t in tasks:
        t.out_degree = len(t.successors) 
    
    calculate_urv(tasks)
    
    # 2. Files de priorité
    odq = sorted(tasks, key=lambda x: (x.out_degree, x.urv), reverse=True) 
    rq = sorted(tasks, key=lambda x: x.urv, reverse=True) 
    
    high_degree_ids = {t.id for t in odq[:l_idx + 1]} 
    
    # Structures pour le suivi du planning
    task_finish_times = {} # {task_id: ft}
    proc_available_time = {p: 0 for p in processors} # Temps de dispo par processeur
    allocation = {} # {task_id: proc_id}

    # 3. Allocation (Boucle principale)
    for task in rq:
        best_proc = None
        earliest_finish_time = float('inf')
        min_energy = float('inf')
        
        # Calcul du Earliest Start Time (EST) basé sur les prédécesseurs
        # Formule (318): max{ft_j + w_ji} [cite: 318]
        def get_est(t, p):
            ready_time = 0
            for pred in t.predecessors:
                comm_delay = pred.comm_costs.get(t.id, 0) if allocation[pred.id] != p else 0
                ready_time = max(ready_time, task_finish_times[pred.id] + comm_delay)
            return max(ready_time, proc_available_time[p])

        if task.id in high_degree_ids:
            # Priorité Performance (Finish Time) [cite: 302, 309]
            for p in processors:
                ft = get_est(task, p) + task.execution_times[p]
                # On simplifie ici sans le paramètre de fiabilité theta
                if ft < earliest_finish_time:
                    earliest_finish_time = ft
                    best_proc = p
        else:
            # Priorité Énergie [cite: 303, 313]
            for p in processors:
                if task.energy_costs[p] < min_energy:
                    min_energy = task.energy_costs[p]
                    best_proc = p
            earliest_finish_time = get_est(task, best_proc) + task.execution_times[best_proc]

        # Enregistrement
        allocation[task.id] = best_proc
        task_finish_times[task.id] = earliest_finish_time
        proc_available_time[best_proc] = earliest_finish_time

    return allocation, task_finish_times