#!/bin/bash
# ./start_vllm.sh
# Launch vLLM server for Qwen2.5-7B-Instruct on A100 (80GB)
# gpu-memory-utilization=0.5 leaves room for faster-whisper on the same GPU

vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8015 \
    --dtype float16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.5 \
    --tensor-parallel-size 1
