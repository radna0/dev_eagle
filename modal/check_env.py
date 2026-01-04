import modal

vllm_image = modal.Image.from_registry(
    "nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11"
).run_commands(
    "pip install torch --extra-index-url https://download.pytorch.org/whl/cu121"
)

app = modal.App("check-env")


@app.function(image=vllm_image, gpu="H100")
def check_env():
    import torch

    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    import subprocess

    subprocess.run(["nvcc", "--version"])


@app.local_entrypoint()
def main():
    check_env.remote()
