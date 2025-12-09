import modal
from pathlib import Path

VERL_REPO_PATH: Path = Path("/root/verl")
image = (
    modal.Image.from_registry("verlai/verl:app-verl0.6-transformers4.56.1-sglang0.5.2-mcore0.13.0-te2.2")
    .apt_install("git")
    .run_commands(f"git clone https://github.com/volcengine/verl {VERL_REPO_PATH}")
    .uv_pip_install("verl[vllm]==0.6.1")
)

app = modal.App("notebook-images-new")

@app.function(image=image)  # You need a Function object to reference the image.
def notebook_image():
    pass