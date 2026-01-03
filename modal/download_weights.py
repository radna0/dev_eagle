import modal
import os

app = modal.App("download-gpt-oss-weights")

# Create a volume to store the weights
run_volume = modal.Volume.from_name("gpt-oss-model-weights", create_if_missing=True)

# Define the image with hf_transfer for faster downloads
image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(image=image, volumes={"/data": run_volume}, timeout=3600)
def download_models():
    from huggingface_hub import snapshot_download

    models = ["openai/gpt-oss-120b", "nvidia/gpt-oss-120b-Eagle3-long-context"]

    for model_id in models:
        print(f"Downloading {model_id}...")
        local_dir = f"/data/{model_id}"
        os.makedirs(local_dir, exist_ok=True)

        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=[
                "original/*",
                "metal/*",
                "*.pt",
                "*.bin",
            ],
        )
        print(f"Successfully downloaded {model_id} to {local_dir}")

    run_volume.commit()
    print("Volume committed. Weights are ready.")


@app.local_entrypoint()
def main():
    download_models.remote()
