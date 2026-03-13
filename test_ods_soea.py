import unittest
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from ods_soea import Task, calculate_urv, get_ods_scheduling
from heft import get_heft_scheduling, calculate_uprank
from datasets import (
    get_dataset_homogeneous_5tasks, 
    get_dataset_heterogeneous_10tasks,
    load_datasets_from_json
)


def draw_task_dag(tasks, title="Task DAG"):
    """
    Dessine le graphe acyclique dirigé (DAG) des tâches avec leurs dépendances.
    Les flèches vont du parent (prédécesseur) au fils (successeur).
    Disposition hiérarchique: père à gauche, fils à droite, frères triés de haut en bas par ordre croissant.
    
    Args:
        tasks: Liste des tâches
        title: Titre du graphe
    """
    # Créer le graphe networkx
    G = nx.DiGraph()
    
    # Ajouter les nœuds
    for task in tasks:
        avg_exec_time = sum(task.execution_times.values()) / len(task.execution_times)
        G.add_node(task.id, label=f"Task {task.id}\n(URV: {task.urv:.1f})\nExec: {avg_exec_time:.1f}")
    
    # Ajouter les arêtes (du prédécesseur au successeur)
    for task in tasks:
        for succ in task.successors:
            comm_cost = task.comm_costs.get(succ.id, 0)
            G.add_edge(task.id, succ.id, weight=comm_cost, label=f"{comm_cost}")
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Disposition hiérarchique manuelle
    def compute_hierarchical_layout(dag_tasks):
        """
        Calcule une disposition hiérarchique du DAG:
        - Père à gauche, fils à droite
        - Frères triés de haut en bas par ordre croissant (ID)
        """
        # Calculer les niveaux topologiques (hauteur dans le DAG)
        task_dict = {task.id: task for task in dag_tasks}
        
        # Calculer la profondeur (niveau) de chaque nœud
        def get_level(task_id, memo={}):
            if task_id in memo:
                return memo[task_id]
            
            task = task_dict[task_id]
            if not task.predecessors:
                level = 0
            else:
                level = 1 + max(get_level(pred.id, memo) for pred in task.predecessors)
            
            memo[task_id] = level
            return level
        
        # Grouper les nœuds par niveau
        levels = {}
        for task in dag_tasks:
            level = get_level(task.id)
            if level not in levels:
                levels[level] = []
            levels[level].append(task.id)
        
        # Trier les nœuds à chaque niveau par ordre croissant (ID)
        for level in levels:
            levels[level].sort()
        
        # Calculer les positions
        pos = {}
        max_level = max(levels.keys())
        
        for level, node_ids in levels.items():
            # Position x: proportionnelle au niveau (gauche à droite)
            x = (level / (max_level + 1)) * 8
            
            # Position y: distribuée verticalement pour les frères
            num_nodes = len(node_ids)
            for i, node_id in enumerate(node_ids):
                # y centré sur 0, échelonné pour éviter les chevauchements
                y = (i - num_nodes / 2 + 0.5) * 2.5
                pos[node_id] = (x, y)
        
        return pos
    
    pos = compute_hierarchical_layout(tasks)
    
    # Dessiner les nœuds
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2000, 
                          alpha=0.9, ax=ax, edgecolors='darkblue', linewidths=2)
    
    # Dessiner les labels des nœuds
    labels_dict = {}
    for task in tasks:
        avg_exec_time = sum(task.execution_times.values()) / len(task.execution_times)
        labels_dict[task.id] = f"T{task.id}\n{avg_exec_time:.1f}"
    nx.draw_networkx_labels(G, pos, labels_dict, 
                           font_size=11, font_weight='bold', ax=ax)
    
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
        task1.comm_costs[task2.id] = 0
        
        self.assertEqual(len(task1.successors), 1)
        self.assertEqual(len(task2.predecessors), 1)
        self.assertEqual(task1.comm_costs[2], 0)


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
        
        task1.comm_costs[task2.id] = 0
        task2.comm_costs[task3.id] = 0
        
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
        
        task1.comm_costs = {task2.id: 0, task3.id: 0}
        task2.comm_costs = {task4.id: 0}
        task3.comm_costs = {task4.id: 0}
        
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
        
        task1.comm_costs[task2.id] = 0
        task2.comm_costs[task3.id] = 0
        
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


# ============================================================================
# CONSOLE INTERACTIVE POUR TESTER L'ALGORITHME ODS
# ============================================================================

def print_header(title):
    """Affiche un titre formaté"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def interactive_scheduler():
    """
    Console interactive pour choisir les paramètres et exécuter le scheduling.
    """
    print_header("SCHEDULING - CONSOLE INTERACTIVE")
    
    # 1. Choix du dataset
    print("\n1. Sélectionner un dataset:")
    print("   [1] Dataset homogène (5 tâches, durées identiques)")
    print("   [2] Dataset hétérogène (10 tâches, durées variables)")
    
    while True:
        choice = input("\nVotre choix (1 ou 2): ").strip()
        if choice in ['1', '2']:
            break
        print("Choix invalide. Veuillez entrer 1 ou 2.")
    
    if choice == '1':
        tasks, default_procs = get_dataset_homogeneous_5tasks()
        dataset_name = "Homogène (5 tâches)"
    else:
        tasks, default_procs = get_dataset_heterogeneous_10tasks()
        dataset_name = "Hétérogène (10 tâches)"
    
    print(f"\n✓ Dataset sélectionné: {dataset_name}")
    print(f"  - Nombre de tâches: {len(tasks)}")
    print(f"  - Processeurs disponibles: {default_procs}")
    
    # 1.5. Choix de l'algorithme
    print("\n1.5. Sélectionner l'algorithme de scheduling:")
    print("   [1] ODS (Optimization Decentralized Scheduling)")
    print("   [2] HEFT (Heterogeneous Earliest Finish Time)")
    
    while True:
        algo_choice = input("\nVotre choix (1 ou 2): ").strip()
        if algo_choice in ['1', '2']:
            break
        print("Choix invalide. Veuillez entrer 1 ou 2.")
    
    if algo_choice == '1':
        algorithm = 'ODS'
    else:
        algorithm = 'HEFT'
    
    print(f"\n✓ Algorithme sélectionné: {algorithm}")
    
    # 2. Nombre de processeurs
    print("\n2. Sélectionner le nombre de processeurs:")
    print(f"   Processeurs disponibles: {default_procs} (max: {len(default_procs)})")
    
    while True:
        try:
            num_procs = int(input(f"Nombre de processeurs (1-{len(default_procs)}): ").strip())
            if 1 <= num_procs <= len(default_procs):
                processors = default_procs[:num_procs]
                break
            else:
                print(f"Veuillez entrer un nombre entre 1 et {len(default_procs)}")
        except ValueError:
            print("Veuillez entrer un nombre valide")
    
    print(f"\n✓ Processeurs sélectionnés: {processors}")
    
    # 3. Paramètres (selon l'algorithme)
    if algorithm == 'ODS':
        # 3. Paramètre l (indices hauts degrés)
        print("\n3. Paramètre 'l' (indice de séparation pour les files de priorité):")
        print(f"   Max: {len(tasks) - 1}")
        
        while True:
            try:
                l_idx = int(input("Valeur de l (0 par défaut): ").strip() or "0")
                if 0 <= l_idx < len(tasks):
                    break
                else:
                    print(f"Veuillez entrer un nombre entre 0 et {len(tasks) - 1}")
            except ValueError:
                print("Veuillez entrer un nombre valide")
        
        print(f"\n✓ Paramètre l: {l_idx}")
        
        # 4. Paramètre theta (fiabilité)
        print("\n4. Paramètre 'theta' (paramètre de fiabilité):")
        
        while True:
            try:
                theta = float(input("Valeur de theta (0.0 par défaut): ").strip() or "0.0")
                if 0 <= theta <= 1:
                    break
                else:
                    print("Veuillez entrer une valeur entre 0 et 1")
            except ValueError:
                print("Veuillez entrer un nombre valide")
        
        print(f"\n✓ Paramètre theta: {theta}")
    else:
        # HEFT n'a pas de paramètres spécifiques
        l_idx = 0
        theta = 0
    
    # 5. Résumé des choix
    print_header("RÉSUMÉ DE LA CONFIGURATION")
    print(f"\n   Algorithme: {algorithm}")
    print(f"   Dataset: {dataset_name}")
    print(f"   Tâches: {len(tasks)}")
    print(f"   Processeurs: {processors}")
    if algorithm == 'ODS':
        print(f"   Paramètre l: {l_idx}")
        print(f"   Paramètre theta: {theta}")
    
    # 6. Exécution du scheduling
    print_header("EXÉCUTION DU SCHEDULING")
    
    # Préparation des tâches selon l'algorithme
    print(f"\nAllocation des tâches avec {algorithm}...")
    
    if algorithm == 'ODS':
        # Recalculer URV et out_degree avec les tâches actuelles
        calculate_urv(tasks)
        for task in tasks:
            task.out_degree = len(task.successors)
        # Exécution ODS
        allocation, finish_times = get_ods_scheduling(tasks, processors, theta=theta, l_idx=l_idx)
    else:  # HEFT
        # Recalculer uprank
        calculate_uprank(tasks)
        # Exécution HEFT
        allocation, finish_times = get_heft_scheduling(tasks, processors, theta=0, l_idx=0)
    
    makespan = max(finish_times.values()) if finish_times else 0
    
    print(f"✓ Allocation terminée")
    print(f"\n   Résultats:")
    print(f"   - Makespan (temps total): {makespan}")
    print(f"   - Tâches allouées: {len(allocation)}")
    
    print(f"\n   Distribution par processeur:")
    for proc_id in processors:
        tasks_on_proc = [tid for tid, p in allocation.items() if p == proc_id]
        print(f"      Processeur {proc_id}: {len(tasks_on_proc)} tâche(s)")
    
    # 7. Visualisations
    print_header("VISUALISATIONS")
    
    print("\nAffichage du DAG des tâches...")
    print("(Fermer la fenêtre pour continuer)")
    draw_task_dag(tasks, f"DAG des Tâches - {dataset_name}")
    
    print("\nAffichage du diagramme de Gantt...")
    print("(Fermer la fenêtre pour continuer)")
    draw_scheduling_gantt(
        allocation, 
        finish_times, 
        tasks, 
        processors, 
        f"Scheduling {algorithm} - {dataset_name}\nMakespan: {makespan}"
    )
    
    # 8. Détail du scheduling
    print_header("DÉTAIL DU SCHEDULING")
    
    for proc_id in sorted(processors):
        tasks_on_proc = sorted(
            [tid for tid, p in allocation.items() if p == proc_id],
            key=lambda tid: finish_times[tid]
        )
        
        if tasks_on_proc:
            print(f"\nProcesseur {proc_id}:")
            for tid in tasks_on_proc:
                ft = finish_times[tid]
                exec_time = tasks[tid].execution_times[proc_id]
                st = ft - exec_time
                print(f"   Task {tid}: [{st:6.0f} - {ft:6.0f}] (durée: {exec_time})")
    
    print_header("FIN")
    print("\nScheduling terminé!")


if __name__ == '__main__':
    import sys
    
    # Vérifier si on doit exécuter les tests ou la console interactive
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Mode test unitaire
        unittest.main(argv=sys.argv[:1])
    else:
        # Mode console interactive
        try:
            interactive_scheduler()
        except KeyboardInterrupt:
            print("\n\nProgramme interrompu par l'utilisateur.")
        except Exception as e:
            print(f"\nErreur: {e}")
            import traceback
            traceback.print_exc()
