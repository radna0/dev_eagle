import modal
import os
import subprocess
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any

# Modal configuration
app = modal.App("benchmark-tir-sync")

# Force using a single H100
gpu_type = "H100"
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.10",
    )
    .pip_install(
        "vllm==0.13.0",
        "pandas",
        "numpy",
        "openai",
        "transformers",
        "openai-harmony>=0.0.3",
        "mcp",
        "jupyter_client",
        "ipykernel",
        "msgspec",
        "jinja2",
        "tiktoken",
    )
    # Overlay modified source files for async tool calling
    .add_local_dir(
        "/home/kojoe/CUDA_async/vllm/vllm",
        remote_path="/usr/local/lib/python3.10/site-packages/vllm",
        copy=True,
        ignore=[".git", "__pycache__", "*.pyc"],
    )
    .add_local_file(
        "/home/kojoe/CUDA_async/modal/local_python_tool.py",
        remote_path="/root/local_python_tool.py",
        copy=True,
    )
    .add_local_file(
        "/home/kojoe/CUDA_async/reference.csv",
        remote_path="/root/reference.csv",
        copy=True,
    )
)

# Constants from user request
SEED = 42
MAX_LEN = 65536
K = 8
TEMPERATURE = 1.0
TOP_P = 1.0
MIN_P = 0.02

# Model paths (Volume)
MODEL_PATH = "/root/models/openai/gpt-oss-120b"
DRAFT_MODEL_PATH = "/root/models/nvidia/gpt-oss-120b-Eagle3-throughput"

# Volume for models and logs
model_volume = modal.Volume.from_name("gpt-oss-model-weights")
logs_volume = modal.Volume.from_name("benchmark-logs", create_if_missing=True)


@app.function(
    gpu=gpu_type,
    image=image,
    timeout=3600,
    volumes={"/root/logs": logs_volume, "/root/models": model_volume},
)
def run_sync_benchmark():
    import threading
    import queue
    from openai import OpenAI
    from transformers import set_seed, AutoTokenizer
    from openai_harmony import (
        HarmonyEncodingName,
        load_harmony_encoding,
        Conversation,
        Message,
        Role,
        SystemContent,
        ReasoningEffort,
        RenderConversationConfig,
        Author,
        TextContent,
    )
    from local_python_tool import PythonTool, add_libs, ensure_last_print

    # 1. Setup Logging
    log_file = "/root/logs/benchmark_tir_sync.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log(msg):
        with open(log_file, "a") as f:
            t = time.strftime("%H:%M:%S")
            print(f"[{t}] {msg}")
            f.write(f"[{t}] {msg}\n")

    log("Starting Sync TIR Benchmark (GPT-OSS-120B)")
    set_seed(SEED)

    # 2. Start vLLM Server
    command = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL_PATH,
        "--served-model-name",
        "gpt-oss",
        "--tensor-parallel-size",
        "1",
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        "0.95",
        "--max-cudagraph-capture-size",
        "2048",
        "--speculative-config",
        f'{{"model":"{DRAFT_MODEL_PATH}","num_speculative_tokens":1,"method":"eagle3","draft_tensor_parallel_size":1}}',
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--dtype",
        "auto",
        "--max-model-len",
        str(MAX_LEN),
        "--stream-interval",
        "20",
    ]

    # Start vLLM server with output streaming to console
    import threading

    vllm_proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # Line buffered
        start_new_session=True,
    )

    # Background thread to stream vLLM server output to console
    def stream_output():
        for line in vllm_proc.stdout:
            print(f"[vLLM] {line}", end="", flush=True)

    output_thread = threading.Thread(target=stream_output, daemon=True)
    output_thread.start()

    # Monitor server startup
    log("Waiting for vLLM server to start...")
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="sk-local")

    for i in range(30):  # 15 minutes max
        try:
            client.models.list()
            log("vLLM server is READY.")
            break
        except Exception as e:
            log(f"Server not ready yet (attempt {i+1}/30): {e}")
            time.sleep(30)
    else:
        log("vLLM server failed to start after 15 minutes.")
        vllm_proc.terminate()
        return

    # 3. Initialize Inferencer Components
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()
    python_pool = queue.Queue(maxsize=K)
    for _ in range(K):
        t = PythonTool(execution_backend="jupyter", local_jupyter_timeout=60.0)
        python_pool.put(t)
    log(f"Python Tool Pool (size={K}) initialized.")

    # 4. Define Inferencer (Sync Mode)
    class SyncInferencer:
        def __init__(self):
            self.client = client
            self.encoding = encoding
            self.stop_token_ids = stop_token_ids

        def generate_tir(self, problem: str, seed_offset: int):
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
            start_gen = time.perf_counter()

            for iter in range(10):  # Max 10 turns
                prompt_ids = self.encoding.render_conversation_for_completion(
                    Conversation.from_messages(messages), Role.ASSISTANT
                )

                # We use non-streaming call for simplicity in sync benchmark if preferred,
                # but the user requested Harmony protocol style which usually streams items.
                # However, for wall-clock benchmark, let's use the core loop from reference.

                response = self.client.completions.create(
                    model="gpt-oss",
                    prompt=prompt_ids,
                    max_tokens=2048,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    seed=SEED + seed_offset,
                    extra_body=dict(
                        min_p=MIN_P,
                        stop_token_ids=self.stop_token_ids,
                        return_token_ids=True,
                    ),
                    timeout=120,
                )

                token_ids = response.choices[0].token_ids

                # DEBUG: Log token IDs and decoded text
                log(f"   [DEBUG] Received {len(token_ids)} tokens")
                log(f"   [DEBUG] First 20 token IDs: {token_ids[:20]}")
                log(f"   [DEBUG] Last 10 token IDs: {token_ids[-10:]}")

                # Decode tokens to see actual text
                decoded_tokens = []
                for i, tid in enumerate(token_ids[:30]):  # First 30 tokens
                    try:
                        text = self.encoding.tokenizer.decode([tid])
                        decoded_tokens.append(f"{tid}:'{text}'")
                    except:
                        decoded_tokens.append(f"{tid}:?")
                log(f"   [DEBUG] First 30 decoded tokens: {decoded_tokens}")

                # In sync mode, parser handles the one-shot completion
                new_msgs = self.encoding.parse_messages_from_completion_tokens(
                    token_ids, Role.ASSISTANT
                )

                # DEBUG: Log parsed messages
                log(f"   [DEBUG] Parsed {len(new_msgs)} messages")
                for i, msg in enumerate(new_msgs):
                    log(
                        f"   [DEBUG] Message {i}: channel={msg.channel}, recipient={msg.recipient}, content_len={len(msg.content) if msg.content else 0}"
                    )

                messages.extend(new_msgs)

                last_msg = messages[-1]
                log(
                    f"   [DEBUG] Last message: channel={last_msg.channel}, recipient={last_msg.recipient}"
                )

                if last_msg.channel == "final" or token_ids[-1] == 200002:  # End token
                    break

                if last_msg.recipient == "python":
                    log(f"   [SYNC] Turning to Python tool (offset={seed_offset})...")
                    py_start = time.perf_counter()
                    # tool logic
                    code = last_msg.content[0].text
                    output = python_tool.execute_code(code)
                    py_elapsed = time.perf_counter() - py_start
                    total_tool_wait += py_elapsed

                    # Tool response
                    output_msg = Message(
                        author=Author(role=Role.TOOL, name="functions.python"),
                        content=[TextContent(text=output)],
                    ).with_recipient("assistant")
                    messages.append(output_msg)

            python_pool.put(python_tool)
            total_elapsed = time.perf_counter() - start_gen
            return messages, total_elapsed, total_tool_wait

    # 5. Run Benchmark against first 10 questions
    df = pd.read_csv("/root/reference.csv")
    target_ids = ["92ba6a", "26de63", "42d360"]
    test_questions = df[df["id"].isin(target_ids)]

    results = []
    sync_inf = SyncInferencer()

    global_start = time.perf_counter()

    for idx, row in test_questions.iterrows():
        problem_id = row["id"]
        question = row["problem"]
        ground_truth = row["answer"]

        log(f"\n--- Question {idx+1}: {problem_id} ---")

        # We run K parallel samples as requested
        q_start = time.perf_counter()

        # Threaded parallel sampling
        parallel_results = [None] * K

        def sample_worker(k_idx):
            try:
                parallel_results[k_idx] = sync_inf.generate_tir(question, k_idx)
            except Exception as e:
                log(f"   [ERROR] Worker {k_idx} failed: {e}")
                parallel_results[k_idx] = ([], 0.0, 0.0)  # Empty result

        threads = []
        for k in range(K):
            t = threading.Thread(target=sample_worker, args=(k,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        q_elapsed = time.perf_counter() - q_start

        # Aggregate stats
        total_q_tool_wait = sum(r[2] for r in parallel_results)
        log(
            f"Completed Question {idx+1} in {q_elapsed:.2f}s (Total Tool Wait: {total_q_tool_wait:.2f}s across K={K})"
        )

        results.append(
            {
                "id": problem_id,
                "latency": q_elapsed,
                "tool_wait": total_q_tool_wait / K,  # average per sample
                "success": False,  # We'd need to parse boxed answers for full accuracy check
            }
        )

    global_total = time.perf_counter() - global_start
    log(f"\n{'='*40}\nSYNC BENCHMARK COMPLETE\n{'='*40}")
    log(f"Total Wall-Clock Time: {global_total:.2f}s")
    log(f"Avg Latency per Question (K={K}): {global_total/10:.2f}s")

    # 6. Cleanup
    vllm_proc.terminate()
    return results


@app.local_entrypoint()
def main():
    run_sync_benchmark.remote()
