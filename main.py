from json_parser import load_tasks_from_json

def main():
    # Define processors
    processors = ["p1", "p2", "p3"]

    # Load tasks from JSON
    tasks = load_tasks_from_json("task_graph_9_8_seed_38872.json", processors)

    # Print loaded tasks to verify
    for task_id, task in tasks.items():
        print(f"Task: {task_id}")
        print(f"  Execution times: {task.execution_times}")
        print(f"  Energy costs: {task.energy_costs}")
        print()

if __name__ == "__main__":
    main()