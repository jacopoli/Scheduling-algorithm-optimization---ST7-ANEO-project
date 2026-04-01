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

def save_results_to_json(tasks, allocation, finish_times, frequencies, proc_map, output_path="results.json"):
    results = []
    for task_id, proc_id in allocation.items():
        f = frequencies[task_id]
        ft = finish_times[task_id]
        results.append({
            "task_id": task_id,
            "processor": proc_id,
            "finish_time": round(ft, 4),
            "frequency": round(f, 6),
        })

    # sum of every tasks energy consumption at max frequency (f=1)
    energy_at_fmax = sum(
        proc_map[allocation[task.id]].energy(1.0, task.execution_times[allocation[task.id]])
        for task in tasks
    )
    energy_soea = sum(
        proc_map[allocation[task.id]].energy(frequencies[task.id], task.execution_times[allocation[task.id]])
        for task in tasks
    )
    # convert savings to percentage
    savings_pct = (energy_at_fmax - energy_soea) / energy_at_fmax * 100

    with open(output_path, "w") as f:
        json.dump({
            "makespan": round(max(finish_times.values()), 4),
            "energy_savings_percentage": round(savings_pct, 2),
            "tasks": results
        }, f, indent=4)

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