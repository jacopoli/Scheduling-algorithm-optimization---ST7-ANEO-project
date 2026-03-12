from json_parser import load_tasks_from_json

def main():
    # Define processors
    processors = ["p1", "p2", "p3"]

    # Load tasks from JSON
    tasks = load_tasks_from_json("task_graph_10_8_seed_76236.json", processors)

    # Print loaded tasks to verify
    for task in tasks:
        print(task)

if __name__ == "__main__":
    main()