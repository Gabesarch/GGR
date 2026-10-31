#!/usr/bin/env python3
"""Serve the Hugging Face backfill teacher and visual judge with vLLM."""
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='Hugging Face model ID or downloaded snapshot')
    parser.add_argument('--port', type=int)
    parser.add_argument('--tensor-parallel-size', type=int)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    parser.add_argument('--dry-run', action='store_true')
    args, extra = parser.parse_known_args()
    model = args.model or 'Qwen/Qwen3.5-122B-A10B'
    command = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', model,
               '--served-model-name', model, '--host', '127.0.0.1',
               '--port', str(args.port or 8000),
               '--tensor-parallel-size', str(args.tensor_parallel_size or 4),
               '--max-model-len', '32000', '--dtype', 'bfloat16',
               '--gpu-memory-utilization', str(args.gpu_memory_utilization),
               '--enforce-eager', '--gdn-prefill-backend', 'triton',
               '--no-enable-prefix-caching', *extra]
    if args.dry_run:
        import shlex
        print(shlex.join(command))
    else:
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
