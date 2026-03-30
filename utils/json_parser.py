import json
from ods import Task
from soea import Processor

def load_tasks_from_json(file_path, processors, comm_cost=1):
    """
    Convert a JSON task file into Task dictionary usable by ODS scheduling.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    tasks_raw = data["tasks"]
    tasks = {}

    # create all tasks
    for t in tasks_raw:
        exec_times = {p: t["duration"] for p in processors}
        energy_costs = {p: t["duration"] * 2 for p in processors}
        task = Task(t["id"], exec_times, energy_costs)
        for tid in t["dependencies"]:
            task.add_predecessor(tasks[tid])
            tasks[tid].add_successor(task)

        tasks[t["id"]] = task

    return list(tasks.values())

def save_results_to_json(allocation, finish_times, output_path="results.json"):
    results = {
        "allocation": allocation,
        "finish_times": finish_times
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

def load_processors_from_json(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    proc_map = {}
    for p in data["processors"]:
        proc = Processor(
            id=p["id"],
            lambda_f=p["lambda_f"],
            d=p["d"],
            f_min=p["f_min"],
            f_max=p["f_max"],
            alpha=p["alpha"],
            c=p["c"],
            p_static=p["p_static"]
        )
        proc_map[proc.id] = proc
    return proc_map