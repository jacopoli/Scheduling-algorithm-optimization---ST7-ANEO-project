import json
from ods_soea import Task

def load_tasks_from_json(file_path, processors, comm_cost=1):
    """
    Convert a JSON task file into Task dictionary usable by ODS scheduling.
    """

    with open(file_path, "r") as f:
        data = json.load(f)

    tasks_raw = data["tasks"]
    tasks = {}
    
    # Create Task objects
    for t in tasks_raw:
        exec_times = {p: t["duration"] for p in processors} # homogeneous processors
        energy_costs = {p: t["duration"] * 2 for p in processors}  # simple model

        task = Task(t["id"], exec_times, energy_costs)
        tasks[t["id"]] = task

    return tasks