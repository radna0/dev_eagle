import modal
import os

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .run_commands(
        "pip install torch==2.9.0 --extra-index-url https://download.pytorch.org/whl/cu128"
    )
    .run_commands(
        "pip install flashinfer-python==0.5.3 flashinfer-cubin==0.5.3 flashinfer-jit-cache==0.5.3 --extra-index-url https://flashinfer.ai/whl/cu128"
    )
    .run_commands("pip install pybind11 ninja wheel setuptools")
    .run_commands(f"echo 'Cache breaker {os.urandom(4).hex()}'")
    .add_local_dir(
        "/home/kojoe/CUDA_eagle/vllm_eagle",
        remote_path="/root/vllm_eagle",
        copy=True,
    )
    .run_commands(
        "FLASHINFER_INCLUDE=$(find /usr/local/lib -name sampling.cuh | head -n 1 | xargs dirname | xargs dirname) && "
        "cd /root/vllm_eagle && FLASHINFER_INCLUDE=$FLASHINFER_INCLUDE TORCH_CUDA_ARCH_LIST='8.0;8.6;8.9;9.0' CUDA_HOME=/usr/local/cuda MAX_JOBS=1 CC=g++ CXX=g++ pip install -v . --no-build-isolation"
    )
)

app = modal.App("test-vllm-eagle")


@app.function(image=vllm_image)
def test_import():
    try:
        import vllm_eagle

        print("Successfully imported vllm_eagle")
        from vllm_eagle import _C

        print("Successfully imported vllm_eagle._C")
        print(f"Available ops in _C: {dir(_C)}")
    except Exception as e:
        print(f"Import failed: {e}")
        import traceback

        traceback.print_exc()


@app.local_entrypoint()
def main():
    print("DEBUG: Calling test_import.remote()...")
    test_import.remote()
    print("DEBUG: Done.")
