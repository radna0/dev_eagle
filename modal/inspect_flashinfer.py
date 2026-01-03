import modal

app = modal.App("flashinfer-inspector")

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .run_commands("pip install --upgrade pip")
    .run_commands(
        "pip install flashinfer-python==0.5.3 flashinfer-cubin==0.5.3 flashinfer-jit-cache==0.5.3 --extra-index-url https://flashinfer.ai/whl/cu128"
    )
)


@app.function(image=image)
def inspect_module():
    import flashinfer

    try:
        import flashinfer.sampling

        print("--- flashinfer.sampling attributes ---")
        print(dir(flashinfer.sampling))
    except ImportError:
        print("Could not import flashinfer.sampling")

    print("\n--- flashinfer attributes ---")
    print(dir(flashinfer))


@app.local_entrypoint()
def main():
    inspect_module.remote()
