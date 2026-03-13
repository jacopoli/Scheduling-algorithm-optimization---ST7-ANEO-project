from matplotlib import pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches

def draw_task_dag(tasks, title="Task DAG", output_path=None):
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
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
    
    plt.close()


def draw_scheduling_gantt(allocation,
                          finish_times,
                          tasks,
                          processors,
                          title="Scheduling Gantt Chart",
                          output_path=None
                          ):
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

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
    
    plt.close()
