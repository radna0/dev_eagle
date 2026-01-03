import modal
import os
import subprocess
import time
import pandas as pd
import numpy as np
import threading
import queue
import json
from typing import List, Dict, Any

# Modal configuration
app = modal.App("tir-phase2-optimized-only")

# Path constants
VLLM_DRIFT_SRC = "/home/kojoe/CUDA_eagle/vllm-drift"
VLLM_EAGLE_SRC = "/home/kojoe/CUDA_eagle/vllm_eagle"
LOCAL_PYTHON_TOOL = "/home/kojoe/CUDA_eagle/local_python_tool.py"
REFERENCE_CSV = "/home/kojoe/CUDA_eagle/reference.csv"

# Model paths on Volume
TARGET_MODEL = "/data/openai/gpt-oss-120b"
LONG_CONTEXT_DRAFT = "/data/nvidia/gpt-oss-120b-Eagle3-long-context"

# Benchmarking constants
MAX_MODEL_LEN = 32768
GPU_MEMORY_UTILIZATION = 0.9
K_SAMPLES = 4
TEMPERATURE = 1.0
MIN_P = 0.02
MAX_TOKENS = 2048

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        "git",
        "ninja-build",
        "pkg-config",
        "libgoogle-perftools-dev",
        "build-essential",
        "clang",
    )
    .pip_install(
        "torch",
        "vllm==0.13.0",
        "transformers",
        "huggingface_hub",
        "hf_transfer",
        "numpy",
        "pandas",
        "polars",
        "datasets",
        "scipy",
        "sentencepiece",
        "protobuf",
        "msgspec",
        "pybind11",
        "ninja",
        "wheel",
        "setuptools",
        "openai",
        "openai-harmony>=0.0.3",
        "jupyter_client",
        "ipykernel",
        "mcp",
        "tiktoken",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PAGER": "cat"})
    .add_local_dir(
        VLLM_DRIFT_SRC, remote_path="/root/vllm-drift-src", copy=True, ignore=[".git"]
    )
    .add_local_dir(
        VLLM_EAGLE_SRC, remote_path="/root/vllm-eagle-src", copy=True, ignore=[".git"]
    )
    .add_local_file(
        LOCAL_PYTHON_TOOL, remote_path="/root/local_python_tool.py", copy=True
    )
    .add_local_file(REFERENCE_CSV, remote_path="/root/reference.csv", copy=True)
    .run_commands(
        "cd /root/vllm-eagle-src && TORCH_CUDA_ARCH_LIST='8.0;8.6;8.9;9.0' MAX_JOBS=1 pip install . --no-build-isolation"
    )
)

volume = modal.Volume.from_name("gpt-oss-model-weights")
logs_volume = modal.Volume.from_name("benchmark-logs", create_if_missing=True)


@app.function(
    gpu="H100",
    image=image,
    volumes={"/data": volume, "/root/logs": logs_volume},
    timeout=7200,
)
def run_tir_benchmark(
    config_name: str, draft_model_path: str, num_spec: int, use_phase2: bool
):
    import os
    import subprocess
    import sys

    print(f"\n{'='*80}")
    print(f"STARTING CONFIG: {config_name}")
    print(f"Draft Model: {draft_model_path}")
    print(f"Num Spec: {num_spec}")
    print(f"Phase 2 Optimized: {use_phase2}")
    print(f"{'='*80}\n")

    # 1. Patch vLLM recursively
    print("Patching vLLM recursively from vllm-drift-src...")
    vllm_path = "/usr/local/lib/python3.11/site-packages/vllm"
    src_path = "/root/vllm-drift-src"
    subprocess.run(f"cp -rv {src_path}/* {vllm_path}/", shell=True, check=True)
    subprocess.run(
        f"find {vllm_path} -type d -name '__pycache__' -exec rm -rf {{}} + 2>/dev/null || true",
        shell=True,
    )
    print("Patch complete.\n")

    # Set environment for Phase 2
    os.environ["VLLM_EAGLE_PHASE2_FUSED"] = "1" if use_phase2 else "0"
    os.environ["VLLM_EAGLE_DRAFT_SAMPLING"] = "1"

    import vllm
    from vllm import EngineArgs, LLMEngine, SamplingParams
    from openai import OpenAI
    from transformers import set_seed
    from openai_harmony import (
        HarmonyEncodingName,
        load_harmony_encoding,
        Conversation,
        Message,
        Role,
        SystemContent,
        ReasoningEffort,
        Author,
        TextContent,
    )
    from local_python_tool import PythonTool

    set_seed(42)

    port = 8000

    server_cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        TARGET_MODEL,
        "--served-model-name",
        "gpt-oss",
        "--tensor-parallel-size",
        "1",
        "--max-num-seqs",
        "10",
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--max-cudagraph-capture-size",
        "1024",
        "--speculative-config",
        f'{{"model":"{draft_model_path}","num_speculative_tokens":{num_spec},"method":"eagle3","draft_tensor_parallel_size":1}}',
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--trust-remote-code",
    ]

    vllm_proc = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )

    def stream_output():
        for line in vllm_proc.stdout:
            print(f"[{config_name}] {line}", end="", flush=True)

    threading.Thread(target=stream_output, daemon=True).start()

    # Wait for server
    client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="sk-local")
    print(f"Waiting for vLLM server on port {port}...")
    for _ in range(60):  # Increase wait time to 30 mins (60 * 30s)
        try:
            client.models.list()
            print("Server is READY.")
            break
        except:
            time.sleep(30)
    else:
        print("Server failed to start.")
        os.killpg(os.getpgid(vllm_proc.pid), 9)
        return {"error": "Server failed"}

    # 3. TIR Logic
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()

    python_pool = queue.Queue()
    for _ in range(K_SAMPLES):
        python_pool.put(PythonTool(execution_backend="jupyter"))

    class SyncInferencer:
        def generate_tir(self, problem: str, k_idx: int):
            python_tool = python_pool.get()
            messages = [
                Message(
                    author=Author(role=Role.SYSTEM, name="system"),
                    content=[
                        SystemContent.new()
                        .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
                        .with_tools(python_tool.tool_config)
                    ],
                ),
                Message(
                    author=Author(role=Role.USER, name="user"),
                    content=[TextContent(text=problem)],
                ),
            ]

            total_tool_wait = 0.0
            total_tokens = 0
            start_gen = time.perf_counter()

            for turn in range(10):  # Max turns
                prompt_ids = encoding.render_conversation_for_completion(
                    Conversation.from_messages(messages), Role.ASSISTANT
                )

                response = client.completions.create(
                    model="gpt-oss",
                    prompt=prompt_ids,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    seed=42 + k_idx,
                    extra_body=dict(
                        min_p=MIN_P,
                        stop_token_ids=stop_token_ids,
                        return_token_ids=True,
                    ),
                    timeout=600,
                )

                token_ids = response.choices[0].token_ids
                total_tokens += len(token_ids)

                new_msgs = encoding.parse_messages_from_completion_tokens(
                    token_ids, Role.ASSISTANT
                )
                messages.extend(new_msgs)

                last_msg = messages[-1]
                if last_msg.channel == "final" or token_ids[-1] == 200002:  # EOT
                    break

                if last_msg.recipient == "python":
                    code = last_msg.content[0].text
                    py_start = time.perf_counter()
                    output = python_tool.execute_code(code)
                    total_tool_wait += time.perf_counter() - py_start

                    output_msg = Message(
                        author=Author(role=Role.TOOL, name="functions.python"),
                        content=[TextContent(text=output)],
                    ).with_recipient("assistant")
                    messages.append(output_msg)

            python_pool.put(python_tool)
            total_time = time.perf_counter() - start_gen
            return messages, total_time, total_tool_wait, total_tokens

    # 4. Run over first 3 questions
    df = pd.read_csv("/root/reference.csv")
    target_ids = ["92ba6a", "26de63", "42d360"]
    test_questions = df[df["id"].isin(target_ids)]

    inferencer = SyncInferencer()
    results = []
    q_logs = []

    for _, row in test_questions.iterrows():
        p_id = row["id"]
        q_text = row["problem"]
        print(f"\nRunning Problem: {p_id}")

        q_start = time.perf_counter()
        parallel_results = [None] * K_SAMPLES

        def worker(i):
            try:
                parallel_results[i] = inferencer.generate_tir(q_text, i)
            except Exception as e:
                print(f"Worker {i} error: {e}")
                parallel_results[i] = (None, 0, 0, 0)

        threads = []
        for i in range(K_SAMPLES):
            t = threading.Thread(target=worker, args=(i,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        q_elapsed = time.perf_counter() - q_start

        # Save logs for quality check
        samples_log = []
        for i, (msgs, wall_time, tool_time, tokens) in enumerate(parallel_results):
            if msgs:
                conv_lines = []
                for m in msgs:
                    role_str = f"{m.author.role}"
                    content_str = ""
                    if m.content:
                        c = m.content[0]
                        if hasattr(c, "text"):
                            content_str = c.text[:500]
                        else:
                            content_str = f"[{type(c).__name__}]"
                    conv_lines.append(f"{role_str}: {content_str}")

                samples_log.append(
                    {
                        "sample_idx": i,
                        "tokens": tokens,
                        "wall_time": wall_time,
                        "tool_time": tool_time,
                        "conv": "\n".join(conv_lines),
                    }
                )

        q_logs.append({"id": p_id, "samples": samples_log})

        results.append(
            {
                "id": p_id,
                "wall_clock": q_elapsed,
                "avg_tokens": sum(r[3] for r in parallel_results if r[3]) / K_SAMPLES,
                "avg_tool_wait": sum(r[2] for r in parallel_results if r[3])
                / K_SAMPLES,
            }
        )

    # Cleanup
    os.killpg(os.getpgid(vllm_proc.pid), 9)

    # Save results to logs volume with unique name
    ts = int(time.time())
    log_file = f"/root/logs/tir_optimized_only_{ts}.json"
    meta_file = f"/root/logs/tir_optimized_only_{ts}_meta.json"

    with open(log_file, "w") as f:
        json.dump({"results": results, "logs": q_logs}, f, indent=2)

    # Calculate global metrics
    total_wall = sum(r["wall_clock"] for r in results)
    total_tokens = sum(r["avg_tokens"] for r in results)
    avg_tps = total_tokens * K_SAMPLES / total_wall if total_wall > 0 else 0

    summary = {
        "config": config_name,
        "total_wall_clock": total_wall,
        "total_avg_tokens": total_tokens,
        "global_tps": avg_tps,
        "timestamp": ts,
    }
    with open(meta_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


@app.local_entrypoint()
def main():
    config = ("Long Context K3 Optimized", LONG_CONTEXT_DRAFT, 3, True)
    name, path, n_spec, p2 = config

    print("\n" + "=" * 80)
    print(f"RUNNING INDIVIDUAL TIR BENCHMARK: {name}")
    print("=" * 80)

    summary = run_tir_benchmark.remote(name, path, n_spec, p2)

    print("\n" + "=" * 80)
    print("BENCHMARK RESULT")
    print("=" * 80)
    if "error" in summary:
        print(f"Error: {summary['error']}")
    else:
        print(f"Config: {summary['config']}")
        print(f"Total Wall-Clock: {summary['total_wall_clock']:.2f}s")
        print(f"Total Avg Tokens: {summary['total_avg_tokens']:.1f}")
        print(f"Global Decode Throughput: {summary['global_tps']:.2f} tok/s")
    print("=" * 80)
