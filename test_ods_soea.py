import unittest
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from ods_soea import Task, calculate_urv, get_ods_scheduling


def draw_task_dag(tasks, title="Task DAG"):
    """
    Dessine le graphe acyclique dirigé (DAG) des tâches avec leurs dépendances.
    Les flèches vont du parent (prédécesseur) au fils (successeur).
    
    Args:
        tasks: Liste des tâches
        title: Titre du graphe
    """
    # Créer le graphe networkx
    G = nx.DiGraph()
    
    # Ajouter les nœuds
    for task in tasks:
        G.add_node(task.id, label=f"Task {task.id}\n(URV: {task.urv:.1f})")
    
    # Ajouter les arêtes (du prédécesseur au successeur)
    for task in tasks:
        for succ in task.successors:
            comm_cost = task.comm_costs.get(succ.id, 0)
            G.add_edge(task.id, succ.id, weight=comm_cost, label=f"{comm_cost}")
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Utiliser une disposition hiérarchique en utilisant le layout spring
    pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
    
    # Si on peut, utiliser une meilleure disposition (graphviz)
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(G, prog='dot')
    except:
        # Fallback sur spring layout avec plus d'espace
        pos = nx.spring_layout(G, k=4, iterations=100, seed=42)
    
    # Dessiner les nœuds
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2000, 
                          alpha=0.9, ax=ax, edgecolors='darkblue', linewidths=2)
    
    # Dessiner les labels des nœuds
    nx.draw_networkx_labels(G, pos, {task.id: f"T{task.id}" for task in tasks}, 
                           font_size=12, font_weight='bold', ax=ax)
    
    # Dessiner les arêtes (flèches) avec plus de visibilité
    nx.draw_networkx_edges(G, pos, edge_color='darkred', arrows=True, 
                          arrowsize=30, arrowstyle='->', width=2.5, 
                          connectionstyle="arc3,rad=0.1", ax=ax, alpha=0.7)
    
    # Ajouter les étiquettes des arêtes (coûts de communication)
    edge_labels = nx.get_edge_attributes(G, 'label')
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10, 
                                     font_weight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                                     facecolor='yellow', alpha=0.7), ax=ax)
    
    # Ajouter une légende explicative
    legend_text = "Flèches: Dépendances (Parent → Fils)\nChiffres: Coûts de communication"
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def draw_scheduling_gantt(allocation, finish_times, tasks, processors, title="Scheduling Gantt Chart"):
    """
    Trace un diagramme de Gantt montrant le scheduling sur les différents processeurs.
    
    Args:
        allocation: Dict {task_id: processor_id}
        finish_times: Dict {task_id: finish_time}
        tasks: Liste des tâches (pour accéder aux temps d'exécution)
        processors: Liste des processeurs
        title: Titre du diagramme
    """
    # Créer un dictionnaire pour accéder aux tâches par ID
    task_dict = {task.id: task for task in tasks}
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Couleurs pour les tâches
    colors = plt.cm.Set3(range(len(allocation)))
    color_map = {task_id: colors[i] for i, task_id in enumerate(sorted(allocation.keys()))}
    
    # Pour chaque processeur et chaque tâche
    for task_id, proc_id in allocation.items():
        task = task_dict[task_id]
        ft = finish_times[task_id]
        exec_time = task.execution_times[proc_id]
        st = ft - exec_time  # Start time
        
        # Dessiner le rectangle pour la tâche
        ax.barh(proc_id, exec_time, left=st, height=0.6, 
                color=color_map[task_id], edgecolor='black', linewidth=1.5)
        
        # Ajouter le label de la tâche
        ax.text(st + exec_time/2, proc_id, f'T{task_id}', 
               ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Configuration des axes
    ax.set_yticks(processors)
    ax.set_yticklabels([f'Processeur {p}' for p in processors])
    ax.set_xlabel('Temps', fontsize=11, fontweight='bold')
    ax.set_ylabel('Processeurs', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Ajouter une grille
    ax.grid(True, axis='x', alpha=0.3)
    
    # Ajouter une légende
    legend_patches = [mpatches.Patch(color=color_map[task_id], label=f'Task {task_id}') 
                     for task_id in sorted(allocation.keys())]
    ax.legend(handles=legend_patches, loc='upper right', ncol=2)
    
    # Limites des axes
    ax.set_xlim(0, max(finish_times.values()) * 1.05)
    ax.set_ylim(-0.5, len(processors) - 0.5)
    
    plt.tight_layout()
    plt.show()


class TestTask(unittest.TestCase):
    """Tests pour la classe Task"""
    
    def test_task_creation(self):
        """Tester la création d'une tâche"""
        execution_times = {0: 10, 1: 12}
        energy_costs = {0: 5, 1: 4}
        task = Task(1, execution_times, energy_costs)
        
        self.assertEqual(task.id, 1)
        self.assertEqual(task.execution_times, execution_times)
        self.assertEqual(task.energy_costs, energy_costs)
        self.assertEqual(task.successors, [])
        self.assertEqual(task.predecessors, [])
        self.assertEqual(task.urv, 0)
        self.assertEqual(task.out_degree, 0)
    
    def test_task_with_dependencies(self):
        """Tester les relations de dépendances entre tâches"""
        task1 = Task(1, {0: 10}, {0: 5})
        task2 = Task(2, {0: 8}, {0: 4})
        
        task1.successors.append(task2)
        task2.predecessors.append(task1)
        task1.comm_costs[task2.id] = 2
        
        self.assertEqual(len(task1.successors), 1)
        self.assertEqual(len(task2.predecessors), 1)
        self.assertEqual(task1.comm_costs[2], 2)


class TestURVCalculation(unittest.TestCase):
    """Tests pour le calcul de l'Up-Rank Value"""
    
    def test_urv_single_task(self):
        """URV pour une tâche sans successeurs"""
        task = Task(1, {0: 10, 1: 12}, {0: 5, 1: 4})
        calculate_urv([task])
        
        # URV = avg_exec_time = (10+12)/2 = 11
        self.assertEqual(task.urv, 11.0)
    
    def test_urv_linear_chain(self):
        """URV pour une chaîne linéaire de tâches"""
        task1 = Task(1, {0: 10}, {0: 5})
        task2 = Task(2, {0: 8}, {0: 4})
        task3 = Task(3, {0: 6}, {0: 3})
        
        # Chaîne: 1 -> 2 -> 3
        task1.successors.append(task2)
        task2.predecessors.append(task1)
        task2.successors.append(task3)
        task3.predecessors.append(task2)
        
        task1.comm_costs[task2.id] = 2
        task2.comm_costs[task3.id] = 2
        
        calculate_urv([task1, task2, task3])
        
        # task3: urv = 6 + 0 = 6
        self.assertEqual(task3.urv, 6)
        
        # task2: urv = 8 + max(2 + 6) = 16
        self.assertEqual(task2.urv, 16)
        
        # task1: urv = 10 + max(2 + 16) = 28
        self.assertEqual(task1.urv, 28)
    
    def test_urv_diamond_dag(self):
        """URV pour un DAG en forme de diamant"""
        # Structure: task1 -> [task2, task3] -> task4
        task1 = Task(1, {0: 10}, {0: 5})
        task2 = Task(2, {0: 8}, {0: 4})
        task3 = Task(3, {0: 7}, {0: 3})
        task4 = Task(4, {0: 6}, {0: 2})
        
        task1.successors = [task2, task3]
        task2.predecessors.append(task1)
        task3.predecessors.append(task1)
        
        task2.successors.append(task4)
        task3.successors.append(task4)
        task4.predecessors = [task2, task3]
        
        task1.comm_costs = {task2.id: 2, task3.id: 2}
        task2.comm_costs = {task4.id: 2}
        task3.comm_costs = {task4.id: 2}
        
        calculate_urv([task1, task2, task3, task4])
        
        # task4: urv = 6
        self.assertEqual(task4.urv, 6)
        
        # task2: urv = 8 + (2 + 6) = 16
        self.assertEqual(task2.urv, 16)
        
        # task3: urv = 7 + (2 + 6) = 15
        self.assertEqual(task3.urv, 15)


class TestODSScheduling(unittest.TestCase):
    """Tests pour l'algorithme ODS de scheduling"""
    
    def test_simple_two_tasks(self):
        """Scheduling simple: 2 tâches indépendantes"""
        task1 = Task(1, {0: 10, 1: 12}, {0: 5, 1: 4})
        task2 = Task(2, {0: 8, 1: 6}, {0: 4, 1: 3})
        
        processors = [0, 1]
        allocation, finish_times = get_ods_scheduling([task1, task2], processors)
        
        # Les deux tâches doivent être allouées
        self.assertIn(1, allocation)
        self.assertIn(2, allocation)
        
        # Les temps de fin doivent être positifs
        self.assertGreater(finish_times[1], 0)
        self.assertGreater(finish_times[2], 0)
    
    def test_linear_chain(self):
        """Scheduling pour une chaîne linéaire de tâches"""
        task1 = Task(1, {0: 10}, {0: 5})
        task2 = Task(2, {0: 8}, {0: 4})
        task3 = Task(3, {0: 6}, {0: 3})
        
        # Dépendances
        task1.successors.append(task2)
        task2.predecessors.append(task1)
        task2.successors.append(task3)
        task3.predecessors.append(task2)
        
        task1.comm_costs[task2.id] = 2
        task2.comm_costs[task3.id] = 2
        
        processors = [0]  # Un seul processeur
        allocation, finish_times = get_ods_scheduling([task1, task2, task3], processors)
        
        # Tous les processeurs doivent être le même
        self.assertEqual(allocation[1], 0)
        self.assertEqual(allocation[2], 0)
        self.assertEqual(allocation[3], 0)
        
        # Les temps de fin doivent respecter les dépendances
        self.assertEqual(finish_times[1], 10)  # 0 + 10
        self.assertEqual(finish_times[2], 18)  # (10 + 2 comm) ou (0 + 8) = max -> 10 + 8 = 18 (même proc)
        self.assertEqual(finish_times[3], 24)  # (18 + 2 comm) ou (0 + 6) = max -> 18 + 6 = 24 (même proc)
    
    def test_parallel_tasks_multiple_processors(self):
        """Scheduling avec tâches parallèles sur plusieurs processeurs"""
        task1 = Task(1, {0: 10, 1: 12}, {0: 5, 1: 4})
        task2 = Task(2, {0: 8, 1: 6}, {0: 4, 1: 3})
        task3 = Task(3, {0: 5, 1: 4}, {0: 2, 1: 1})
        
        # Pas de dépendances
        processors = [0, 1]
        allocation, finish_times = get_ods_scheduling([task1, task2, task3], processors)
        
        # Chaque tâche doit être allouée
        self.assertEqual(len(allocation), 3)
        
        # Les processeurs doivent avoir été utilisés
        used_procs = set(allocation.values())
        self.assertTrue(len(used_procs) > 0)
    
    def test_out_degree_calculation(self):
        """Tester que l'out_degree est correctement calculé"""
        task1 = Task(1, {0: 10}, {0: 5})
        task2 = Task(2, {0: 8}, {0: 4})
        task3 = Task(3, {0: 6}, {0: 3})
        
        # task1 a 2 successeurs
        task1.successors = [task2, task3]
        task2.predecessors.append(task1)
        task3.predecessors.append(task1)
        
        task1.comm_costs[task2.id] = 0
        task1.comm_costs[task3.id] = 0
        
        # task2 et task3 n'ont pas de successeurs
        
        processors = [0]
        allocation, finish_times = get_ods_scheduling([task1, task2, task3], processors)
        
        # task1 devrait avoir out_degree = 2
        self.assertEqual(task1.out_degree, 2)
        self.assertEqual(task2.out_degree, 0)
        self.assertEqual(task3.out_degree, 0)
    
    def test_empty_task_list(self):
        """Scheduling avec liste vide de tâches"""
        processors = [0, 1]
        allocation, finish_times = get_ods_scheduling([], processors)
        
        self.assertEqual(len(allocation), 0)
        self.assertEqual(len(finish_times), 0)
    
    def test_heterogeneous_execution_times(self):
        """Scheduling avec des coûts d'exécution hétérogènes"""
        # task1 est rapide sur proc 1
        task1 = Task(1, {0: 100, 1: 10}, {0: 50, 1: 5})
        # task2 est rapide sur proc 0
        task2 = Task(2, {0: 8, 1: 80}, {0: 4, 1: 40})
        
        processors = [0, 1]
        allocation, finish_times = get_ods_scheduling([task1, task2], processors)
        
        # task1 devrait être sur le proc qui minimise le temps (haut out_degree avec URV)
        # task2 devrait être sur le proc qui minimise l'énergie (bas out_degree)
        self.assertIn(allocation[1], processors)
        self.assertIn(allocation[2], processors)


class TestSchedulingProperties(unittest.TestCase):
    """Tests des propriétés du scheduling"""
    
    def test_no_task_overlap_on_processor(self):
        """Vérifier qu'il n'y a pas de chevauchement des tâches sur un processeur"""
        task1 = Task(1, {0: 10}, {0: 5})
        task2 = Task(2, {0: 8}, {0: 4})
        task3 = Task(3, {0: 6}, {0: 3})
        
        processors = [0]
        allocation, finish_times = get_ods_scheduling([task1, task2, task3], processors)
        
        # Chaque tâche sur le même processeur ne doit pas avoir de chevauchement
        # (pas de vérification théorique avec dependencies)
        total_time = sum(task.execution_times[0] for task in [task1, task2, task3])
        max_finish = max(finish_times.values())
        self.assertGreaterEqual(max_finish, total_time)
    
    def test_all_tasks_allocated(self):
        """Vérifier que toutes les tâches sont allouées"""
        tasks = [
            Task(i, {0: 10-i, 1: 12-i}, {0: 5, 1: 4})
            for i in range(1, 6)
        ]
        
        processors = [0, 1]
        allocation, finish_times = get_ods_scheduling(tasks, processors)
        
        # Toutes les tâches doivent être allouées
        self.assertEqual(len(allocation), 5)
        self.assertEqual(len(finish_times), 5)
        
        for i in range(1, 6):
            self.assertIn(i, allocation)
            self.assertIn(i, finish_times)


class TestVerboseScheduling(unittest.TestCase):
    """Test verbose pour comprendre chaque étape de l'algorithme ODS"""
    
    def test_verbose_scheduling_example(self):
        """
        Cas detaillé: DAG en diamant avec 4 tâches et 2 processeurs
        Structure: task0 -> [task1, task2] -> task3
        """
        print("\n" + "="*80)
        print("EXEMPLE DE SCHEDULING ODS - DAG EN DIAMANT")
        print("="*80)
        
        # Création des tâches
        task0 = Task(0, {0: 10, 1: 12}, {0: 50, 1: 40})
        task1 = Task(1, {0: 8, 1: 9}, {0: 30, 1: 25})
        task2 = Task(2, {0: 7, 1: 8}, {0: 28, 1: 22})
        task3 = Task(3, {0: 6, 1: 7}, {0: 20, 1: 18})
        
        # Définir le DAG: task0 -> task1, task0 -> task2, task1 -> task3, task2 -> task3
        task0.successors = [task1, task2]
        task1.predecessors = [task0]
        task2.predecessors = [task0]
        
        task1.successors = [task3]
        task2.successors = [task3]
        task3.predecessors = [task1, task2]
        
        # Coûts de communication
        task0.comm_costs = {task1.id: 2, task2.id: 2}
        task1.comm_costs = {task3.id: 3}
        task2.comm_costs = {task3.id: 3}
        
        processors = [0, 1]
        
        print("\n1. STRUCTURE DES TÂCHES")
        print("-" * 80)
        tasks = [task0, task1, task2, task3]
        for task in tasks:
            print(f"   Task {task.id}:")
            print(f"      - Temps d'exécution: {task.execution_times}")
            print(f"      - Coûts d'énergie: {task.energy_costs}")
            print(f"      - Successeurs: {[s.id for s in task.successors]}")
            print(f"      - Prédécesseurs: {[p.id for p in task.predecessors]}")
        
        print("\n2. CALCUL DE URV (Up-Rank Value)")
        print("-" * 80)
        from ods_soea import calculate_urv
        calculate_urv(tasks)
        
        for task in tasks:
            print(f"   Task {task.id}: URV = {task.urv}")
        
        print("\n3. CALCUL DE OUT-DEGREE")
        print("-" * 80)
        for task in tasks:
            task.out_degree = len(task.successors)
            print(f"   Task {task.id}: out_degree = {task.out_degree}")
        
        print("\n4. PRIORITÉS")
        print("-" * 80)
        odq = sorted(tasks, key=lambda x: (x.out_degree, x.urv), reverse=True)
        print(f"   ODS Queue (par out_degree puis URV): {[t.id for t in odq]}")
        
        rq = sorted(tasks, key=lambda x: x.urv, reverse=True)
        print(f"   Ready Queue (par URV): {[t.id for t in rq]}")
        
        print("\n5. ALLOCATION DES TÂCHES")
        print("-" * 80)
        allocation, finish_times = get_ods_scheduling(tasks, processors)
        
        for task_id in sorted(allocation.keys()):
            proc = allocation[task_id]
            ft = finish_times[task_id]
            print(f"   Task {task_id} -> Processeur {proc}, Finish Time = {ft}")
        
        print("\n6. RÉSUMÉ DU SCHEDULING")
        print("-" * 80)
        print(f"   Processeur 0: {[tid for tid, p in allocation.items() if p == 0]}")
        print(f"   Processeur 1: {[tid for tid, p in allocation.items() if p == 1]}")
        print(f"   Makespan (temps total): {max(finish_times.values())}")
        
        print("\n7. ORDONNANCEMENT FINAL")
        print("-" * 80)
        for proc_id in processors:
            tasks_on_proc = sorted(
                [tid for tid, p in allocation.items() if p == proc_id],
                key=lambda tid: finish_times[tid]
            )
            print(f"   Processeur {proc_id}:")
            for tid in tasks_on_proc:
                exec_time = task0.execution_times.get(proc_id) if tid == 0 else \
                           task1.execution_times.get(proc_id) if tid == 1 else \
                           task2.execution_times.get(proc_id) if tid == 2 else \
                           task3.execution_times.get(proc_id)
                print(f"      - Task {tid}: Finish Time = {finish_times[tid]}")
        
        print("\n" + "="*80)
        
        # Assertions pour valider le test
        self.assertEqual(len(allocation), 4)
        self.assertEqual(len(finish_times), 4)
        self.assertTrue(all(tid in allocation for tid in range(4)))
        self.assertTrue(all(allocation[tid] in processors for tid in range(4)))
        
        # Afficher les graphes
        print("\n8. VISUALISATIONS")
        print("-" * 80)
        print("   Affichage du DAG des tâches...")
        draw_task_dag(tasks, "DAG des Tâches")
        
        print("   Affichage du diagramme de Gantt...")
        draw_scheduling_gantt(allocation, finish_times, tasks, processors, 
                            f"Scheduling ODS - Makespan: {max(finish_times.values())}")


    def test_verbose_complex_dag(self):
        """
        Cas complexe: DAG multi-niveaux avec 8 tâches et 3 processeurs hétérogènes.
        Structure multi-niveaux avec différentes branches et coûts hétérogènes.
        """
        print("\n" + "="*80)
        print("EXEMPLE COMPLEXE DE SCHEDULING ODS - DAG MULTI-NIVEAUX (8 TÂCHES)")
        print("="*80)
        
        # Création de 8 tâches avec durées hétérogènes
        tasks = [
            Task(0, {0: 15, 1: 18, 2: 12}, {0: 80, 1: 70, 2: 65}),     # Tâche source
            Task(1, {0: 8, 1: 6, 2: 10}, {0: 40, 1: 35, 2: 45}),       # Rapide sur proc 1
            Task(2, {0: 12, 1: 20, 2: 10}, {0: 50, 1: 60, 2: 42}),     # Rapide sur proc 2
            Task(3, {0: 10, 1: 15, 2: 8}, {0: 45, 1: 55, 2: 38}),      # Rapide sur proc 2
            Task(4, {0: 7, 1: 9, 2: 11}, {0: 35, 1: 42, 2: 48}),       # Rapide sur proc 0
            Task(5, {0: 14, 1: 12, 2: 16}, {0: 70, 1: 65, 2: 75}),     # Rapide sur proc 1
            Task(6, {0: 9, 1: 11, 2: 8}, {0: 42, 1: 50, 2: 40}),       # Rapide sur proc 2
            Task(7, {0: 6, 1: 8, 2: 7}, {0: 30, 1: 40, 2: 35}),        # Tâche finale
        ]
        
        # Structure complexe du DAG:
        # Level 0: Task 0
        # Level 1: Task 0 -> [Task 1, Task 2, Task 3]
        # Level 2: Task 1 -> Task 4, Task 2 -> [Task 5, Task 6], Task 3 -> Task 6
        # Level 3: [Task 4, Task 5, Task 6] -> Task 7
        
        tasks[0].successors = [tasks[1], tasks[2], tasks[3]]
        for i in [1, 2, 3]:
            tasks[i].predecessors = [tasks[0]]
        
        tasks[1].successors = [tasks[4]]
        tasks[4].predecessors = [tasks[1]]
        
        tasks[2].successors = [tasks[5], tasks[6]]
        tasks[5].predecessors = [tasks[2]]
        tasks[6].predecessors = [tasks[2]]
        
        tasks[3].successors = [tasks[6]]
        tasks[6].predecessors.append(tasks[3])
        
        tasks[4].successors = [tasks[7]]
        tasks[5].successors = [tasks[7]]
        tasks[6].successors = [tasks[7]]
        tasks[7].predecessors = [tasks[4], tasks[5], tasks[6]]
        
        # Définir les coûts de communication
        tasks[0].comm_costs = {1: 2, 2: 3, 3: 2}
        tasks[1].comm_costs = {4: 2}
        tasks[2].comm_costs = {5: 3, 6: 2}
        tasks[3].comm_costs = {6: 2}
        tasks[4].comm_costs = {7: 2}
        tasks[5].comm_costs = {7: 3}
        tasks[6].comm_costs = {7: 2}
        
        processors = [0, 1, 2]
        
        print("\n1. STRUCTURE DES TÂCHES (DAG Complexe)")
        print("-" * 80)
        print("Niveau 0: Task 0 (source)")
        print("Niveau 1: Task 0 → [Task 1, Task 2, Task 3]")
        print("Niveau 2: Task 1 → Task 4")
        print("         Task 2 → [Task 5, Task 6]")
        print("         Task 3 → Task 6")
        print("Niveau 3: [Task 4, Task 5, Task 6] → Task 7 (sink)")
        print()
        
        for task in tasks:
            succ_str = [s.id for s in task.successors]
            print(f"   Task {task.id}:")
            print(f"      - Temps d'exécution: {task.execution_times}")
            print(f"      - Coûts d'énergie: {task.energy_costs}")
            print(f"      - Successeurs: {succ_str}")
        
        print("\n2. CALCUL DE URV (Up-Rank Value)")
        print("-" * 80)
        calculate_urv(tasks)
        
        for task in sorted(tasks, key=lambda x: x.urv, reverse=True):
            print(f"   Task {task.id}: URV = {task.urv:.2f}")
        
        print("\n3. CALCUL DE OUT-DEGREE")
        print("-" * 80)
        for task in tasks:
            task.out_degree = len(task.successors)
            print(f"   Task {task.id}: out_degree = {task.out_degree}")
        
        print("\n4. PRIORITÉS")
        print("-" * 80)
        odq = sorted(tasks, key=lambda x: (x.out_degree, x.urv), reverse=True)
        print(f"   ODS Queue (par out_degree puis URV): {[t.id for t in odq]}")
        
        rq = sorted(tasks, key=lambda x: x.urv, reverse=True)
        print(f"   Ready Queue (par URV): {[t.id for t in rq]}")
        
        print("\n5. ALLOCATION DES TÂCHES")
        print("-" * 80)
        allocation, finish_times = get_ods_scheduling(tasks, processors)
        
        for task_id in sorted(allocation.keys()):
            proc = allocation[task_id]
            ft = finish_times[task_id]
            task_obj = tasks[task_id]
            exec_time = task_obj.execution_times[proc]
            st = ft - exec_time
            print(f"   Task {task_id}: Proc {proc}, Start={st:5.0f}, Finish={ft:5.0f}, Exec={exec_time}")
        
        print("\n6. RÉSUMÉ DU SCHEDULING")
        print("-" * 80)
        for proc_id in processors:
            tasks_on_proc = sorted(
                [tid for tid, p in allocation.items() if p == proc_id],
                key=lambda tid: finish_times[tid]
            )
            print(f"   Processeur {proc_id}: Tasks {tasks_on_proc}")
        
        makespan = max(finish_times.values())
        total_energy = sum(tasks[tid].energy_costs[allocation[tid]] for tid in range(len(tasks)))
        print(f"\n   Makespan (temps total): {makespan}")
        print(f"   Énergie totale: {total_energy}")
        
        print("\n7. DÉTAIL DE L'ORDONNANCEMENT")
        print("-" * 80)
        for proc_id in processors:
            tasks_on_proc = sorted(
                [tid for tid, p in allocation.items() if p == proc_id],
                key=lambda tid: finish_times[tid]
            )
            if tasks_on_proc:
                print(f"   Processeur {proc_id}:")
                for tid in tasks_on_proc:
                    ft = finish_times[tid]
                    exec_time = tasks[tid].execution_times[proc_id]
                    st = ft - exec_time
                    print(f"      └─ Task {tid}: [{st:5.0f} - {ft:5.0f}] (durée: {exec_time})")
        
        print("\n" + "="*80)
        
        # Assertions
        self.assertEqual(len(allocation), 8)
        self.assertEqual(len(finish_times), 8)
        self.assertTrue(all(tid in allocation for tid in range(8)))
        self.assertTrue(all(allocation[tid] in processors for tid in range(8)))
        
        # Afficher les graphes
        print("\n8. VISUALISATIONS")
        print("-" * 80)
        print("   Affichage du DAG complexe des tâches...")
        draw_task_dag(tasks, "DAG Complexe - 8 Tâches Multi-Niveaux")
        
        print("   Affichage du diagramme de Gantt...")
        draw_scheduling_gantt(allocation, finish_times, tasks, processors, 
                            f"Scheduling ODS Complexe - Makespan: {makespan}")


if __name__ == '__main__':
    unittest.main()
