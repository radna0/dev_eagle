import modal

app = modal.App("flashinfer-debugger-v2")

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .run_commands("pip install --upgrade pip")
    .run_commands(
        "pip install flashinfer-python==0.5.3 flashinfer-cubin==0.5.3 flashinfer-jit-cache==0.5.3 --extra-index-url https://flashinfer.ai/whl/cu128"
    )
)


@app.function(image=image)
def inspect_enums():
    import subprocess
    import os

    def run(cmd):
        print(f"Running: {cmd}")
        try:
            subprocess.run(cmd, shell=True, check=False)
        except Exception as e:
            print(f"Error: {e}")

    print("--- SEARCHING FOR BlockScanAlgorithm definition ---")
    # Search for definition of BlockScanAlgorithm
    run(
        'grep -r "enum .*BlockScanAlgorithm" /usr/local/lib/python3.11/site-packages/flashinfer'
    )

    print("\n--- SEARCHING FOR kSglangWarpScan ---")
    run('grep -r "kSglangWarpScan" /usr/local/lib/python3.11/site-packages/flashinfer')

    print("\n--- LISTING .cuh FILES IN FLASHINFER ---")
    run('find /usr/local/lib/python3.11/site-packages/flashinfer -name "*.cuh"')

    print("\n--- DUMPING layout.cuh (common place for enums) IF EXISTS ---")
    # Just a guess, looking for files that might contain the enum
    run(
        'grep -l -r "BlockScanAlgorithm" /usr/local/lib/python3.11/site-packages/flashinfer | xargs cat'
    )


@app.local_entrypoint()
def main():
    inspect_enums.remote()
