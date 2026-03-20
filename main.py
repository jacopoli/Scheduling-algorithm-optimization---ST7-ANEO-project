import sys
import json
from json_parser import load_tasks_from_json, save_results_to_json
from ods_soea import get_ods_scheduling
from test_ods_soea import draw_scheduling_gantt
from visualizer import draw_task_dag
import time

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <num_processors>")
        sys.exit(1)

    num_processors = int(sys.argv[1])
    processors = [f"p{i}" for i in range(num_processors)]

    # Load tasks from JSON
    tasks = load_tasks_from_json("task_graph.json", processors)
    start = time.time()
    allocation, finish_times = get_ods_scheduling(tasks, processors, l_idx=len(tasks) - 1)
    end = time.time()
    print(f"Algorithm execution time: {end - start:.8f} seconds")

    # Dump results to JSON
    save_results_to_json(allocation, finish_times)

    # Save Gantt chart
    draw_task_dag(tasks, output_path="graph.png", task_prefix="")
    draw_scheduling_gantt(allocation, finish_times, tasks, processors, output_path="gantt.png", task_prefix="")

if __name__ == "__main__":
    main()