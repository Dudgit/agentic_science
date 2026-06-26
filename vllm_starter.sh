#!/bin/bash

#conda activate vllm082
#VLLM_WORKER_MULTIPROC_METHOD=spawn NCCL_P2P_DISABLE=1 vllm serve Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 2 --disable-custom-all-reduce --enforce-eager
VLLM_WORKER_MULTIPROC_METHOD=spawn NCCL_P2P_DISABLE=1 vllm serve Qwen/Qwen2.5-Coder-32B-Instruct --tensor-parallel-size 8 --disable-custom-all-reduce --enforce-eager