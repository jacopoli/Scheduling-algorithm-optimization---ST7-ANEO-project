import json
import sys
from utils.json_parser import load_tasks_from_json
from test_ods_soea import draw_scheduling_gantt
from utils.visualizer import draw_task_dag

def main():
    if len(sys.argv) < 4:
        print("Usage: python generate_plots.py <results.json> <task_graph.json> <num_processors>")
        sys.exit(1)

    results_file = sys.argv[1]
    task_graph_file = sys.argv[2]
    num_processors = int(sys.argv[3])

    processors = [f"p{i}" for i in range(num_processors)]

    # Load results
    with open(results_file, "r") as f:
        results = json.load(f)
    allocation = results["allocation"]
    finish_times = results["finish_times"]

    # Load tasks
    tasks = load_tasks_from_json(task_graph_file, processors)

    # Generate plots
    draw_task_dag(tasks, output_path="graph.png", task_prefix="")
    draw_scheduling_gantt(allocation, finish_times, tasks, processors, output_path="gantt.png", task_prefix="")
    print("Plots saved to graph.png and gantt.png")

if __name__ == "__main__":
    main()