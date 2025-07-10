import wandb
import os

# Perform login once at the beginning of your script.
# Make sure this API key is valid for your self-hosted instance and has access
# to the specified project.
WANDB_HOST = "https://api.wandb.ai"
WANDB_API_KEY = "9cbd211ef6297b8a0644114acad7810f6f7ef1e4"

wandb.login(host="https://api.wandb.ai", key="9cbd211ef6297b8a0644114acad7810f6f7ef1e4")

def fetch_model_from_wandb(url: str) -> str:
    """Download a model artifact from W&B or return local path.

    If ``url`` is a local file path or ``file://`` URL, it is returned as-is.
    Otherwise the artifact is downloaded via the W&B API using the key from
    WANDB_API_KEY and connecting to the specified WANDB_HOST.
    The path to the downloaded model file is returned.
    """
    # Support file:// URLs and plain local paths for offline tests
    if url.startswith("file://"):
        local = url[7:]
        if os.path.exists(local):
            return local
    if os.path.exists(url):
        return url

    # Initialize the W&B API client.
    # By default, it will use the host and API key set by wandb.login()
    # However, for clarity and robustness with self-hosted instances,
    # you can optionally pass the host explicitly if you encounter issues.
    api = wandb.Api() # Explicitly pass the host to the API client
    print(api.settings['base_url'])  # This should print your self-hosted W&B URL
    # Ensure the URL has a version tag (e.g., ':latest')
    if ":" not in url:
        url = f"{url}:latest"

    try:
        # The artifact name format for api.artifact() is "entity/project/artifact_name:version"
        # Your input "davi/smart-home-intent-classifier/smarthome-clf-v1:latest" matches this.
        artifact = api.artifact(url)
    except wandb.errors.CommError as e:
        print(f"Error fetching artifact: {e}")
        print(f"Attempted URL: {url}")
        print(f"Please ensure the project 'smart-home-intent-classifier' exists under entity 'davi' on {WANDB_HOST}")
        print(f"Also verify your API key's permissions for this project.")
        raise

    path = artifact.download()

    # Try to locate a Keras model file inside the downloaded directory
    # (This part of your original logic is good)
    for fname in os.listdir(path):
        if fname.endswith(".keras") or fname.endswith(".h5"):
            return os.path.join(path, fname)
    return path

# Call the function with your artifact path
load_model = fetch_model_from_wandb("davidiogenes51-ufrn/smart-home-intent-classifier/smarthome-clf-v1")
print(f"Model downloaded to: {load_model}")