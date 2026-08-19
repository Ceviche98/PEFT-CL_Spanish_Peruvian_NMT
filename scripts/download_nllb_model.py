import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("[ERROR] huggingface_hub is not installed. Installing right now...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub"])
    from huggingface_hub import snapshot_download

def main():
    project_root = Path(__file__).resolve().parent.parent
    target_dir = project_root / "models" / "nllb-200-1.3B"
    target_dir.mkdir(parents=True, exist_ok=True)

    token = "YOUR_HF_TOKEN_HERE"
    repo_id = "facebook/nllb-200-1.3B"

    print(f"======================================================================")
    print(f"Downloading '{repo_id}' directly to: {target_dir}")
    print(f"======================================================================")
    print("Using token authentication for high-speed download...")

    try:
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            token=token,
            # Ignore git / large training repo metadata not needed for inference/training
            ignore_patterns=["*.git*", "*.msgpack", "flax_model*", "tf_model*", "*.ot"]
        )
        print(f"\n[SUCCESS] Model downloaded cleanly and ready at: {downloaded_path}")
        print("You can now zip or upload this folder ('models/nllb-200-1.3B') directly to Google Drive!")
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
