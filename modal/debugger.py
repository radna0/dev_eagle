import modal

app = modal.App("flashinfer-debugger")

# Same base image steps as dynamic_eagle_benchmark.py
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .run_commands("pip install --upgrade pip")
    .run_commands(
        "pip install flashinfer-python==0.5.3 flashinfer-cubin==0.5.3 flashinfer-jit-cache==0.5.3 --extra-index-url https://flashinfer.ai/whl/cu128"
    )
)


@app.function(image=image)
def inspect_headers():
    import subprocess
    import os

    # helper
    def run(cmd):
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True, check=False)

    print("--- FINDING sampling.cuh ---")
    run("find /usr/local/lib -name sampling.cuh")

    # Read the file if found
    try:
        path = (
            subprocess.check_output(
                "find /usr/local/lib -name sampling.cuh | head -n 1", shell=True
            )
            .decode()
            .strip()
        )
        if path:
            print(f"Found at: {path}")
            print(f"--- CONTENT OF {path} ---")
            run(f"cat {path}")

            # Also look for where BlockScanAlgorithm is defined
            # It's likely in a file included by sampling.cuh
            # Let's grep for kSglangWarpScan in the whole system
            print("--- SEARCHING for kSglangWarpScan ---")
            run(f"grep -r 'kSglangWarpScan' /usr/local/lib")

            print("--- SEARCHING for BlockScanAlgorithm ---")
            run(
                f"grep -r 'BlockScanAlgorithm' /usr/local/lib/python3.11/site-packages/flashinfer | head -n 20"
            )
        else:
            print("sampling.cuh not found!")

    except subprocess.CalledProcessError:
        print("Error during search")


@app.local_entrypoint()
def main():
    inspect_headers.remote()
