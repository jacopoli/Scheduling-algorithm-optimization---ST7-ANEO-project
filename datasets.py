"""
Module contenant les datasets pour les tests de l'algorithme ODS de scheduling.
Fournit deux datasets distincts avec leurs représentations JSON.
Les données JSON sont stockées dans le dossier 'datasets/' avec deux fichiers:
- datasets/homogeneous.json
- datasets/heterogeneous.json
"""

import json
import os
from ods_soea import Task, calculate_urv


def task_to_dict(task):
    """
    Convertit un objet Task en dictionnaire sérialisable en JSON.
    
    Args:
        task: Objet Task
        
    Returns:
        dict: Représentation JSON-compatible de la tâche
    """
    return {
        'id': task.id,
        'execution_times': task.execution_times,
        'energy_costs': task.energy_costs,
        'successors': [s.id for s in task.successors],
        'predecessors': [p.id for p in task.predecessors],
        'comm_costs': task.comm_costs,
        'urv': task.urv,
        'out_degree': task.out_degree
    }


def dict_to_task(task_dict):
    """
    Recréé un objet Task à partir d'un dictionnaire (provenant d'un JSON).
    NOTE: Les successors/predecessors ne sont initialement que des IDs, 
          ils seront reconnectés après création de toutes les tâches.
    
    Args:
        task_dict: Dictionnaire contenant les données d'une tâche
        
    Returns:
        Task: Objet Task reconstruit
    """
    task = Task(
        task_dict['id'],
        task_dict['execution_times'],
        task_dict['energy_costs']
    )
    task.comm_costs = task_dict['comm_costs']
    task.urv = task_dict['urv']
    task.out_degree = task_dict['out_degree']
    
    return task


def tasks_from_json(json_data):
    """
    Reconstruit une liste de tâches à partir des données JSON.
    
    Args:
        json_data: Liste de dictionnaires (ou string JSON)
        
    Returns:
        list: Liste d'objets Task avec toutes les relations restaurées
    """
    # Convertir JSON string si nécessaire
    if isinstance(json_data, str):
        json_data = json.loads(json_data)
    
    # Créer les tâches
    task_dict = {}
    for task_data in json_data:
        task = dict_to_task(task_data)
        task_dict[task.id] = task
    
    # Restaurer les relations
    for i, task_data in enumerate(json_data):
        task = task_dict[task_data['id']]
        
        # Restaurer les successors
        task.successors = [task_dict[succ_id] for succ_id in task_data['successors']]
        
        # Restaurer les predecessors
        task.predecessors = [task_dict[pred_id] for pred_id in task_data['predecessors']]
    
    # Retourner les tâches triées par ID
    return sorted(task_dict.values(), key=lambda t: t.id)


# ============================================================================
# DATASET 1: HOMOGÈNE (5 TÂCHES)
# ============================================================================

def get_dataset_homogeneous_5tasks():
    """
    Dataset 1: 5 tâches avec durées HOMOGÈNES
    Structure: DAG linéaire simple: Task 0 → Task 1 → Task 2 → Task 3 → Task 4
    Tous les processeurs ont les mêmes temps d'exécution (10 unités) et énergie (5 unités)
    """
    tasks = [
        Task(0, {0: 10, 1: 10}, {0: 5, 1: 5}),
        Task(1, {0: 10, 1: 10}, {0: 5, 1: 5}),
        Task(2, {0: 10, 1: 10}, {0: 5, 1: 5}),
        Task(3, {0: 10, 1: 10}, {0: 5, 1: 5}),
        Task(4, {0: 10, 1: 10}, {0: 5, 1: 5}),
    ]
    
    # Structure linéaire
    tasks[0].successors = [tasks[1]]
    tasks[1].predecessors = [tasks[0]]
    tasks[1].successors = [tasks[2]]
    tasks[2].predecessors = [tasks[1]]
    tasks[2].successors = [tasks[3]]
    tasks[3].predecessors = [tasks[2]]
    tasks[3].successors = [tasks[4]]
    tasks[4].predecessors = [tasks[3]]
    
    # Coûts de communication (tous à 0)
    for task in tasks:
        for succ in task.successors:
            task.comm_costs[succ.id] = 0
    
    # Calculer URV et out_degree
    calculate_urv(tasks)
    for task in tasks:
        task.out_degree = len(task.successors)
    
    return tasks, [0, 1]


# ============================================================================
# DATASET 2: HÉTÉROGÈNE (10 TÂCHES)
# ============================================================================

def get_dataset_heterogeneous_10tasks():
    """
    Dataset 2: 10 tâches avec durées VARIABLES (puissances de 2)
    Durées: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 unités (puissances de 2)
    Structure DAG multi-niveaux complexe avec branches parallèles
    """
    tasks = [
        # Level 0
        Task(0, {0: 8, 1: 8, 2: 8}, {0: 8, 1: 8, 2: 8}),
        
        # Level 1 (parallèle)
        Task(1, {0: 2, 1: 2, 2: 2}, {0: 2, 1: 2, 2: 2}),
        Task(2, {0: 64, 1: 64, 2: 64}, {0: 64, 1: 64, 2: 64}),
        
        # Level 2 (branches divergentes)
        Task(3, {0: 8, 1: 8, 2: 8}, {0: 8, 1: 8, 2: 8}),
        Task(4, {0: 16, 1: 16, 2: 16}, {0: 16, 1: 16, 2: 16}),
        Task(5, {0: 32, 1: 32, 2: 32}, {0: 32, 1: 32, 2: 32}),
        
        # Level 3
        Task(6, {0: 64, 1: 64, 2: 64}, {0: 64, 1: 64, 2: 64}),
        Task(7, {0: 64, 1: 64, 2: 64}, {0: 64, 1: 64, 2: 64}),
        
        # Level 4
        Task(8, {0: 16, 1: 16, 2: 16}, {0: 16, 1: 16, 2: 16}),
        
        # Level 5 (sink)
        Task(9, {0: 128, 1: 128, 2: 128}, {0: 128, 1: 128, 2: 128}),
    ]
    
    # Structure DAG multi-niveaux:
    # 0 → [1, 2]
    # 1 → [3, 4]
    # 2 → [4, 5]
    # 3 → 6
    # 4 → 7
    # 5 → 7
    # 6 → 8
    # 7 → [8, 9]
    # 8 → 9
    
    tasks[0].successors = [tasks[1], tasks[2]]
    tasks[1].predecessors = [tasks[0]]
    tasks[2].predecessors = [tasks[0]]
    
    tasks[1].successors = [tasks[3], tasks[4]]
    tasks[3].predecessors = [tasks[1]]
    tasks[4].predecessors = [tasks[1]]
    
    tasks[2].successors = [tasks[4], tasks[5]]
    tasks[4].predecessors.append(tasks[2])
    tasks[5].predecessors = [tasks[2]]
    
    tasks[3].successors = [tasks[6]]
    tasks[6].predecessors = [tasks[3]]
    
    tasks[4].successors = [tasks[7]]
    tasks[7].predecessors = [tasks[4]]
    
    tasks[5].successors = [tasks[7]]
    tasks[7].predecessors.append(tasks[5])
    
    tasks[6].successors = [tasks[8]]
    tasks[8].predecessors = [tasks[6]]
    
    tasks[7].successors = [tasks[8], tasks[9]]
    tasks[8].predecessors.append(tasks[7])
    tasks[9].predecessors = [tasks[7]]
    
    tasks[8].successors = [tasks[9]]
    tasks[9].predecessors.append(tasks[8])
    
    # Coûts de communication (tous à 0)
    for task in tasks:
        for succ in task.successors:
            task.comm_costs[succ.id] = 0
    
    # Calculer URV et out_degree
    calculate_urv(tasks)
    for task in tasks:
        task.out_degree = len(task.successors)
    
    return tasks, [0, 1, 2]


# ============================================================================
# DONNÉES JSON DES DATASETS
# ============================================================================

DATASETS_FOLDER = os.path.join(os.path.dirname(__file__), 'datasets')

def ensure_datasets_folder_exists():
    """Crée le dossier datasets s'il n'existe pas"""
    if not os.path.exists(DATASETS_FOLDER):
        os.makedirs(DATASETS_FOLDER)


def save_dataset_to_json_file(dataset_name, tasks, processors, description):
    """
    Sauvegarde un dataset dans un fichier JSON.
    
    Args:
        dataset_name: Nom du fichier (sans extension) 'homogeneous' ou 'heterogeneous'
        tasks: Liste des tâches
        processors: Liste des processeurs
        description: Description du dataset
    """
    ensure_datasets_folder_exists()
    
    # Préparer les données
    data = {
        'name': dataset_name.replace('_', ' ').title(),
        'description': description,
        'num_tasks': len(tasks),
        'num_processors': len(processors),
        'processors': processors,
        'tasks': [task_to_dict(t) for t in tasks]
    }
    
    # Sauvegarder
    filepath = os.path.join(DATASETS_FOLDER, f'{dataset_name}.json')
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    return filepath


def load_dataset_from_json_file(dataset_name):
    """
    Charge un dataset depuis un fichier JSON.
    
    Args:
        dataset_name: Nom du fichier (sans extension) 'homogeneous' ou 'heterogeneous'
        
    Returns:
        tuple: (tasks, processors)
    """
    filepath = os.path.join(DATASETS_FOLDER, f'{dataset_name}.json')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    tasks = tasks_from_json(data['tasks'])
    processors = data['processors']
    
    return tasks, processors


def initialize_datasets():
    """
    Initialise et sauvegarde les deux datasets dans le dossier datasets/.
    À exécuter une fois pour créer les fichiers JSON.
    """
    # Dataset homogène
    tasks_homo, procs_homo = get_dataset_homogeneous_5tasks()
    save_dataset_to_json_file(
        'homogeneous',
        tasks_homo,
        procs_homo,
        'DAG linéaire avec 5 tâches de durées identiques (10 unités)'
    )
    
    # Dataset hétérogène
    tasks_hetero, procs_hetero = get_dataset_heterogeneous_10tasks()
    save_dataset_to_json_file(
        'heterogeneous',
        tasks_hetero,
        procs_hetero,
        'DAG multi-niveaux avec 10 tâches de durées variables (puissances de 2)'
    )
    
    print("Datasets initialized in 'datasets/' folder")
    print(f"  - {os.path.join(DATASETS_FOLDER, 'homogeneous.json')}")
    print(f"  - {os.path.join(DATASETS_FOLDER, 'heterogeneous.json')}")


# ============================================================================
# FONCTIONS D'ACCÈS AUX DONNÉES
# ============================================================================

def get_all_datasets_json():
    """
    Retourne un dictionnaire contenant les deux datasets chargés depuis les fichiers JSON.
    
    Returns:
        dict: {
            'homogeneous': {...},
            'heterogeneous': {...}
        }
    """
    ensure_datasets_folder_exists()
    
    # Vérifier si les fichiers existent, sinon les créer
    homo_path = os.path.join(DATASETS_FOLDER, 'homogeneous.json')
    hetero_path = os.path.join(DATASETS_FOLDER, 'heterogeneous.json')
    
    if not os.path.exists(homo_path) or not os.path.exists(hetero_path):
        initialize_datasets()
    
    # Charger les données
    with open(homo_path, 'r') as f:
        homo_data = json.load(f)
    with open(hetero_path, 'r') as f:
        hetero_data = json.load(f)
    
    return {
        'homogeneous': homo_data,
        'heterogeneous': hetero_data
    }


def load_datasets_from_json(json_str_or_dict=None):
    """
    Charge les datasets à partir des fichiers JSON dans le dossier datasets/.
    
    Args:
        json_str_or_dict: (Optionnel) String JSON ou dictionnaire pour charger depuis mémoire
        
    Returns:
        tuple: (homogeneous_tasks, homogeneous_processors, heterogeneous_tasks, heterogeneous_processors)
    """
    if json_str_or_dict is None:
        # Charger depuis les fichiers JSON
        homo_tasks, homo_procs = load_dataset_from_json_file('homogeneous')
        hetero_tasks, hetero_procs = load_dataset_from_json_file('heterogeneous')
    else:
        # Charger depuis mémoire
        if isinstance(json_str_or_dict, str):
            data = json.loads(json_str_or_dict)
        else:
            data = json_str_or_dict
        
        homo_tasks = tasks_from_json(data['homogeneous']['tasks'])
        homo_procs = data['homogeneous']['processors']
        
        hetero_tasks = tasks_from_json(data['heterogeneous']['tasks'])
        hetero_procs = data['heterogeneous']['processors']
    
    return homo_tasks, homo_procs, hetero_tasks, hetero_procs


if __name__ == '__main__':
    # Initialiser les datasets JSON
    print("Initializing datasets...")
    initialize_datasets()
    
    # Exemple d'utilisation
    print("\nEXEMPLE D'UTILISATION DES DATASETS")
    print("=" * 80)
    
    # 1. Récupérer les datasets directement
    print("\n1. Récupérer les datasets directement:")
    tasks1, procs1 = get_dataset_homogeneous_5tasks()
    tasks2, procs2 = get_dataset_heterogeneous_10tasks()
    print(f"   Dataset 1: {len(tasks1)} tâches, {len(procs1)} processeurs")
    print(f"   Dataset 2: {len(tasks2)} tâches, {len(procs2)} processeurs")
    
    # 2. Charger depuis les fichiers JSON
    print("\n2. Charger depuis les fichiers JSON:")
    h_tasks, h_procs, he_tasks, he_procs = load_datasets_from_json()
    print(f"   Chargées: {len(h_tasks)} tâches (homo) et {len(he_tasks)} tâches (hetero)")
    
    # 3. Accéder aux données via dictionnaire JSON
    print("\n3. Accéder aux données via dictionnaire JSON:")
    all_datasets = get_all_datasets_json()
    print(f"   Datasets disponibles: {list(all_datasets.keys())}")
    print(f"   Dataset homogène: {all_datasets['homogeneous']['name']}")
    print(f"   Dataset hétérogène: {all_datasets['heterogeneous']['name']}")
    
    # 4. Afficher le chemin des fichiers
    print("\n4. Fichiers créés:")
    print(f"   Dossier: {DATASETS_FOLDER}")
    for fname in ['homogeneous.json', 'heterogeneous.json']:
        fpath = os.path.join(DATASETS_FOLDER, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            print(f"   - {fname} ({size} bytes)")
