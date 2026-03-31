import json
import boto3
from utils.json_parser import load_tasks_from_json, save_results_to_json
from ods import get_ods_scheduling

s3 = boto3.client('s3')

def handler(event, context):
    num_processors = event.get("num_processors", 3)
    input_bucket = event.get("input_bucket")
    input_key = event.get("input_key", "task_graph.json")
    output_bucket = event.get("output_bucket", input_bucket)

    # Download input file from S3 to /tmp
    local_input = "/tmp/task_graph.json"
    s3.download_file(input_bucket, input_key, local_input)

    # Run scheduling
    processors = [f"p{i}" for i in range(num_processors)]
    tasks = load_tasks_from_json(local_input, processors)
    allocation, finish_times = get_ods_scheduling(tasks, processors, l_idx=len(tasks) - 1)

    # Save results to /tmp
    local_results = "/tmp/results.json"
    save_results_to_json(allocation, finish_times, output_path=local_results)

    # Upload results to S3
    prefix = input_key.replace(".json", "")
    s3.upload_file(local_results, output_bucket, f"{prefix}_results.json")

    return {
        "statusCode": 200,
        "body": json.dumps({
            "results": f"s3://{output_bucket}/{prefix}_results.json",
            "makespan": max(finish_times.values())
        })
    }