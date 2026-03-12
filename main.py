from json_parser import load_tasks_from_json

def main():
    # Define processors
    processors = ["p1", "p2", "p3"]

    # Load tasks from JSON
    tasks = load_tasks_from_json("task_graph_20_15_seed_36537.json", processors)

    # Print loaded tasks to verify
    for task in tasks:
        print(f"Task: {task.id}")
        print(f"Execution times: {task.execution_times}")
        print(f"Energy costs: {task.energy_costs}")
        print(f"Predecessors: {task.predecessors}")
        print()

if __name__ == "__main__":
    main()