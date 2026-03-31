from soea import soea
from utils.json_parser import load_processors_from_json, load_tasks_from_json, save_results_to_json
from ods import get_ods_scheduling
import time

def main():
    processors = load_processors_from_json("processors.json")
    tasks = load_tasks_from_json("task_graph.json", processors)

    start = time.time()
    allocation, finish_times = get_ods_scheduling(tasks, processors, l_idx=len(tasks) - 1)
    frequencies = soea(tasks, allocation, processors, r_req=0.6)
    end = time.time()
    print(f"Algorithm execution time: {end - start:.8f} seconds")

    # Dump results to JSON
    save_results_to_json(tasks, allocation, finish_times, frequencies, processors)

    # Save Gantt chart
    # draw_task_dag(tasks, output_path="graph.png", task_prefix="")
    # draw_scheduling_gantt(allocation, finish_times, tasks, processors, output_path="gantt.png", task_prefix="")

if __name__ == "__main__":
    main()