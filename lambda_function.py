import json
import boto3
from soea import soea
from utils.json_parser import load_processors_from_json, load_tasks_from_json, save_results_to_json
from ods import get_ods_scheduling

s3 = boto3.client('s3')

def handler(event, context):
    input_bucket = event.get("input_bucket")
    input_graph_key = event.get("input_graph_key", "task_graph.json")
    input_processors_key = event.get("input_processors_key", "processors.json")
    output_bucket = event.get("output_bucket", input_bucket)

    # Download input file from S3 to /tmp
    local_input_graph = "/tmp/task_graph.json"
    s3.download_file(input_bucket, input_graph_key, local_input_graph)

    local_input_processors = "/tmp/processors.json"
    s3.download_file(input_bucket, input_processors_key, local_input_processors)

    # Run scheduling
    processors = load_processors_from_json(local_input_processors)
    tasks = load_tasks_from_json(local_input_graph, processors)
    allocation, finish_times = get_ods_scheduling(tasks, processors, l_idx=len(tasks) - 1)
    frequencies = soea(tasks, allocation, processors, r_req=0.6)
    
    # Save results to /tmp
    local_results = "/tmp/results.json"
    save_results_to_json(tasks, allocation, finish_times, frequencies, processors, output_path=local_results)

    # Upload results to S3
    prefix = input_graph_key.replace(".json", "")
    s3.upload_file(local_results, output_bucket, f"{prefix}_results.json")

    return {
        "statusCode": 200,
        "body": json.dumps({
            "results": f"s3://{output_bucket}/{prefix}_results.json",
            "makespan": max(finish_times.values())
        })
    }