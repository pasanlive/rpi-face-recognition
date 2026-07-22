import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def setup():
    """
    Downloads or symlinks local Hailo-8 model zoo files from DeGirum/hailo_examples
    so the face recognition app can run 100% offline without requiring a cloud token.
    """
    logger.info("=== Raspberry Pi 5 AI Hat+ Face Recognition Model Setup ===")

    # Check if local models folder already exists and has content
    if os.path.exists("./models") and os.path.isdir("./models") and len(os.listdir("./models")) > 0:
        logger.info("Local './models' directory already exists with models. Ready for offline execution!")
        return True

    # Search candidates
    candidates = [
        "../hailo_examples/models",
        "../models",
        os.path.expanduser("~/Documents/hailo_examples/models"),
        os.path.expanduser("~/hailo_examples/models"),
        os.path.expanduser("~/models")
    ]

    for candidate in candidates:
        if os.path.exists(candidate) and os.path.isdir(candidate):
            logger.info(f"Found local models in '{candidate}'. Creating symlink to './models'...")
            try:
                if os.path.islink("./models") or os.path.exists("./models"):
                    os.remove("./models")
                os.symlink(os.path.abspath(candidate), "./models")
                logger.info("Symlink created successfully! Local offline models are ready.")
                return True
            except Exception as e:
                logger.warning(f"Symlink creation failed: {e}")

    # Clone hailo_examples repository automatically
    logger.info("Cloning DeGirum hailo_examples repository to get offline Hailo-8 models...")
    clone_cmd = "git clone --depth 1 https://github.com/DeGirum/hailo_examples.git ../hailo_examples"
    res = subprocess.run(clone_cmd, shell=True)

    target_repo_models = "../hailo_examples/models"
    if os.path.exists(target_repo_models):
        try:
            if os.path.exists("./models"):
                os.remove("./models")
            os.symlink(os.path.abspath(target_repo_models), "./models")
            logger.info("SUCCESS: Downloaded local models and symlinked to './models'!")
            logger.info("The application will now run 100% offline without requiring any cloud token!")
            return True
        except Exception as e:
            logger.error(f"Failed to create symlink: {e}")

    logger.warning("Could not automatically download models. You can clone manually:\n"
                   "  git clone https://github.com/DeGirum/hailo_examples.git ../hailo_examples")
    return False

if __name__ == "__main__":
    setup()
