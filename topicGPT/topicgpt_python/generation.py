import pandas as pd
from topicgpt_python.utils import *
import regex
import traceback
from sentence_transformers import SentenceTransformer, util
import argparse
import os
from anytree import Node, RenderTree
import jsonlines

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def prompt_formatting(generation_prompt, docs, topics_list):
    """
    Simplifies the prompt formatting process.
    """
    topic_str = "\n".join(topics_list)
    combined_docs = "\n\n".join([f"[Document {i+1}]\n{doc}" for i, doc in enumerate(docs)])
    prompt = generation_prompt.format(Topics=topic_str, Document_Batch=combined_docs)
    # Output combined_docs to a jsonl file
    # with jsonlines.open('combined_docs.jsonl', mode='w') as writer:
    #     writer.write({'combined_docs': combined_docs})
    # # Output prompt to a jsonl file
    # with jsonlines.open('prompt.jsonl', mode='w') as writer:
    #     writer.write({'prompt': prompt})
    return prompt

def generate_topics(docs, api_client, generation_prompt, topics_root, topics_list, batch_size):
    """
    Simplified topic generation logic with batch processing.
    """
    responses = []
    topic_format = regex.compile(r"\[1\] (\w+): (.+)")
    for start in range(0, len(docs), batch_size):
        batch_docs = docs[start:start + batch_size]
        prompt = prompt_formatting(generation_prompt, batch_docs, topics_list)
        response = api_client.iterative_prompt(prompt, max_tokens=2000, temperature=0.0, top_p=1.0, verbose=False)
        if response is None:
            response = "[1] None: No topics."
        cnt = 0
        for single_response in response.split("\n\n"):
            cnt += 1
            # print(cnt)
            # print(single_response)
            # Parse each document's response and update topics tree
            matches = regex.findall(topic_format, single_response)
            # if len(matches) > 1:
            #     print(cnt)
            #     print("Multiple topics found:", matches)
            for match in matches:
                lvl = 1
                topic_label, description = match
                # print(match)
                # print(topic_label)
                # existing_node = next((node for node in topics_root.root.children if node.name == topic_label), None)
                dups = topics_root.find_duplicates(topic_label, lvl)
                if (dups):
                    dups[0].count += 1
                else:
                    topics_root._add_node(lvl, topic_label, 1, description, topics_root.root)
                    topics_list = topics_root.to_topic_list(desc=False, count=False)
            responses.append(single_response)
    return responses, topics_list, topics_root

def generate_topic_lvl1(api, model, data_file, prompt_file, seed_file, out_file, topic_file, verbose, batch_size, session_id):
    """
    Entry point for generating high-level topics from documents.
    """
    api_client = APIClient(api=api, model=model)

    df = pd.read_json(data_file, lines=True)
    docs = df["text"].tolist()
    with open(prompt_file, 'r') as file:
        generation_prompt = file.read()

    topics_root = TopicTree().from_seed_file(seed_file)
    topics_list = topics_root.to_topic_list(desc=True, count=False)

    responses, topics_list, topics_root = generate_topics(docs, api_client, generation_prompt, topics_root, topics_list, batch_size)
    topics_root.to_file(topic_file)

    try:
        df = df.iloc[: len(responses)]
        df["responses"] = responses
        df.to_json(out_file, lines=True, orient="records")
    except Exception as e:
        traceback.print_exc()
        with open(f"data/output/generation_1_backup_{model}_{session_id}.txt", "w") as f:
            for line in responses:
                print(line, file=f)

    if verbose:
        print("Generation completed. Results are saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4")
    parser.add_argument("--data_file", type=str, default="data/input/sample.jsonl")
    parser.add_argument("--prompt_file", type=str, default="prompt/generation_1.txt")
    parser.add_argument("--seed_file", type=str, default="prompt/seed_1.md")
    parser.add_argument("--out_file", type=str, default="data/output/generation_1.jsonl")
    parser.add_argument("--topic_file", type=str, default="data/output/generation_1.md")
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=20)
    args = parser.parse_args()

    generate_topic_lvl1(
        args.api,
        args.model,
        args.data_file,
        args.prompt_file,
        args.seed_file,
        args.out_file,
        args.topic_file,
        args.verbose,
        args.batch_size
    )
