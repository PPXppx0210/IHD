import sys
import os
from topicgpt_python import *

def process_data_file(data_file, worker_id):
    log_file = f"logs/worker_{worker_id}.log"
    error_log_file = f"logs/worker_{worker_id}_error.log"
    os.makedirs("logs", exist_ok=True)
    sys.stdout = open(log_file, "w")
    sys.stderr = open(error_log_file, "w")

    print(f"Worker {worker_id} is processsing file: {data_file}")
    output_file = f"data/output/Weibo-Aminer/generation_weibo_{worker_id}.jsonl"
    topic_output_file = f"data/output/Weibo-Aminer/generation_weibo_{worker_id}.md"

    generate_topic_lvl1(
        api = "openai",
        model = "claude-3-5-sonnet-20241022",
        data_file = data_file,
        prompt_file = "prompt/generation.txt",
        seed_file = "prompt/seed.md",
        out_file = output_file,  
        topic_file = topic_output_file,
        verbose = True,
        batch_size = 30,
        session_id = worker_id,
    )

    return f"Worker {worker_id} processing {data_file} finished."

data_file = sys.argv[1]
worker_id = int(sys.argv[2])
process_data_file(data_file, worker_id)