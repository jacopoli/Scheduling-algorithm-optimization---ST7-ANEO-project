import sys
from json_parser import load_tasks_from_json

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <task_graph_file.json>")
        sys.exit(1)

    filename = sys.argv[1]

    # Define processors
    processors = ["p1", "p2", "p3"]

    # Load tasks from JSON
    tasks = load_tasks_from_json(filename, processors)

    # Print loaded tasks to verify
    for task in tasks:
        print(task)

if __name__ == "__main__":
    main()