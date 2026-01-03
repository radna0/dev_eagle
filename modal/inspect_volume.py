import modal

volume = modal.Volume.from_name("gpt-oss-model-weights")
app = modal.App("inspect-volume")


@app.function(volumes={"/data": volume})
def list_data():
    import os

    print("Listing /data:")
    for root, dirs, files in os.walk("/data"):
        depth = root.replace("/data", "").count(os.sep)
        if depth > 3:
            continue
        print(f"{'  ' * depth}[{root}]")
        for f in files[:3]:
            print(f"{'  ' * (depth + 1)}{f}")


@app.local_entrypoint()
def main():
    list_data.remote()
