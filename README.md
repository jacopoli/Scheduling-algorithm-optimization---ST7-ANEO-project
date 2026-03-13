# ODS+SOEA Scheduling Algorithm - Cloud Deployment Guide

## Prerequisites
- AWS CLI installed and configured
- An AWS account with Lambda and S3 permissions

## Setup
> **⚠️ Important**
> If you are logged in and have already created the bucket, go to section 3.
### 1. Configure AWS credentials
```bash
aws configure
```
Enter your AWS Access Key ID, Secret Access Key, region (`us-east-1`) and output file json.

### 2. Create S3 bucket and folders
```bash
# Create bucket
aws s3 mb s3://central-supelec-data-group4 --region us-east-1

# Create input and output folders
aws s3api put-object --bucket central-supelec-data-group4 --key input_data/
aws s3api put-object --bucket central-supelec-data-group4 --key output_data/
```

### 3. Upload your task graph
```bash
aws s3 cp task_graph.json s3://central-supelec-data-group4/input_data/task_graph.json
```
Note: make sure to specify the full filename at the end of the S3 path.

### 4. Package and deploy the Lambda function
> **⚠️ Important**
> If you haven't changed anything in the code, you can skip this section as it is already done

```bash
# Zip the required files
zip my_lambda.zip lambda_function.py ods_soea.py json_parser.py

# Deploy to Lambda
aws lambda create-function \
    --function-name ordonnanceur_group4 \
    --runtime python3.13 \
    --handler lambda_function.handler \
    --zip-file fileb://my_lambda.zip \
    --role arn:aws:iam::<account-id>:role/LambdaS3Role-group4 \
    --region us-east-1
```

Check that the function is ready (State should be `Active`):
```bash
aws lambda get-function --function-name ods_soea --region us-east-1
```

## Running the Algorithm
### Invoke the Lambda function
```bash
aws lambda invoke \
    --function-name ods_soea \
    --payload '{"input_bucket": "central-supelec-data-group4", "input_key": "input_data/task_graph.json", "num_processors": <desired_proc>}' \
    --cli-binary-format raw-in-base64-out \
    response.json \
    --region us-east-1
```

### Check the response
```bash
cat response.json
```

A successful response looks like:
```json
{
    "statusCode": 200,
    "body": {
        "results": "s3://central-supelec-data-group4/input_data/task_graph_results.json",
        "makespan": 285.0
    }
}
```

### Download the results
```bash
aws s3 cp s3://central-supelec-data-group4/input_data/task_graph_results.json results.json
cat results.json
```

## Generating Plots Locally
Once you have `results.json`, generate the Gantt chart and DAG graph locally:
```bash
python generate_plots.py results.json task_graph.json <num_processors>
```

## Input Format
The task graph JSON file must follow this structure:
```json
{
    "graph_id": "my_graph",
    "tasks": [
        {
            "id": "task1",
            "duration": 25,
            "memory": 2048,
            "dependencies": []
        },
        {
            "id": "task2",
            "duration": 22,
            "memory": 256,
            "dependencies": ["task1"]
        }
    ]
}
```

## Parameters
| Parameter        | Description                         | Default                      |
|------------------|-------------------------------------|------------------------------|
| `input_bucket`   | S3 bucket name                      | required                     |
| `input_key`      | Path to input JSON in S3            | `input_data/task_graph.json` |
| `num_processors` | Number of processors to schedule on | `3`                          |
| `output_bucket`  | S3 bucket for results               | same as `input_bucket`       |