import modal

app = modal.App("minimal-test")


@app.function()
def hello():
    print("Hello from Modal!")
    return "success"


@app.local_entrypoint()
def main():
    print(hello.remote())
