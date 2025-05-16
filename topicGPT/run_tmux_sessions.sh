#!/bin/bash

data_files=(
    "data/input/Twitter-Huangxin/chunk_0.jsonl"
    "data/input/Twitter-Huangxin/chunk_1.jsonl"
    "data/input/Twitter-Huangxin/chunk_2.jsonl"
    "data/input/Twitter-Huangxin/chunk_3.jsonl"
)

for idx in {0..3}; do
    data_file="${data_files[$idx]}"
    tmux new-session -d -s "worker_$idx" "python script_tmux_session.py $data_file $idx"
    echo "Start tmux session: worker_$idx, processing file: $data_file"
done

echo "All tmux session start, use 'tmux ls' to check session list."