"""
Module contenant l'implémentation de l'algorithme HEFT (Heterogeneous Earliest Finish Time)
pour le scheduling de tâches sur architectures hétérogènes.

HEFT est un algorithme classique composé de deux phases:
1. Calcul des rangs (uprank) : évalue l'importance critique de chaque tâche
2. Allocation : assigne les tâches aux processeurs en minimisant le temps de fin d'exécution
"""


def calculate_uprank(tasks):
    """
    Calcule l'uprank (up_rank) de façon récursive de droite à gauche dans le DAG.
    
    L'uprank d'une tâche est défini comme:
    uprank(task) = avg_execution_time(task) + max{comm_cost(task, succ) + uprank(succ)}
    
    Args:
        tasks: Liste d'objets Task avec leurs successeurs et coûts de communication
        
    Returns:
        None (met à jour l'attribut uprank de chaque tâche)
    """
    # Traiter les tâches en ordre inverse (de la fin vers le début)
    for task in reversed(tasks):
        avg_exec = sum(task.execution_times.values()) / len(task.execution_times)
        
        max_succ_val = 0
        for succ in task.successors:
            # uprank du successeur + coût de communication
            comm_cost = task.comm_costs.get(succ.id, 0)
            val = comm_cost + succ.uprank
            if val > max_succ_val:
                max_succ_val = val
        
        # uprank de la tâche courante
        task.uprank = avg_exec + max_succ_val


def get_heft_scheduling(tasks, processors, theta=0, l_idx=0):
    """
    Algorithme HEFT (Heterogeneous Earliest Finish Time) pour le scheduling de tâches.
    
    Phase 1: Calcule l'uprank de chaque tâche (importance critique)
    Phase 2: Alloue les tâches dans l'ordre décroissant d'uprank, en choisissant 
             le processeur qui minimise le temps de fin d'exécution
    
    Args:
        tasks: Liste d'objets Task avec leurs propriétés (execution_times, successors, etc.)
        processors: Liste d'identifiants de processeurs
        theta: Paramètre non utilisé (pour compatibilité avec l'interface ODS)
        l_idx: Paramètre non utilisé (pour compatibilité avec l'interface ODS)
        
    Returns:
        tuple: (allocation, task_finish_times)
            - allocation: {task_id: processor_id}
            - task_finish_times: {task_id: finish_time}
    """
    
    # 1. Calculer l'uprank de chaque tâche
    calculate_uprank(tasks)
    
    # 2. Trier les tâches par uprank décroissant
    sorted_tasks = sorted(tasks, key=lambda x: x.uprank, reverse=True)
    
    # Structures pour le suivi du planning
    task_finish_times = {}              # {task_id: ft}
    proc_available_time = {p: 0 for p in processors}  # Temps de disponibilité par processeur
    allocation = {}                      # {task_id: proc_id}
    
    # 3. Boucle d'allocation principale
    for task in sorted_tasks:
        best_proc = None
        earliest_finish_time = float('inf')
        
        # Pour chaque processeur, calculer le Finish Time si on y alloue cette tâche
        for proc in processors:
            # Calculer l'Earliest Start Time (EST)
            # = max de la disponibilité du processeur et du ready time des prédécesseurs
            est = proc_available_time[proc]
            
            # Considérer le ready time des prédécesseurs
            for pred in task.predecessors:
                pred_finish_time = task_finish_times[pred.id]
                
                # Ajouter le coût de communication si le prédécesseur n'est pas sur le même processeur
                if allocation[pred.id] != proc:
                    comm_cost = pred.comm_costs.get(task.id, 0)
                    pred_finish_time += comm_cost
                
                est = max(est, pred_finish_time)
            
            # Calculer le Finish Time = EST + temps d'exécution sur ce processeur
            ft = est + task.execution_times[proc]
            
            # Garder le processeur qui donne le plus petit Finish Time
            if ft < earliest_finish_time:
                earliest_finish_time = ft
                best_proc = proc
        
        # Enregistrer l'allocation
        allocation[task.id] = best_proc
        task_finish_times[task.id] = earliest_finish_time
        proc_available_time[best_proc] = earliest_finish_time
    
    return allocation, task_finish_times
