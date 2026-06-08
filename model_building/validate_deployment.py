from pathlib import Path

DEPLOYMENT_DIR = Path("deployment")

required_files = [
    "app.py",
    "requirements.txt",
    "Dockerfile",
    "README.md",
    "push_to_hf_space.py"
]

missing_files = []

for file_name in required_files:
    file_path = DEPLOYMENT_DIR / file_name
    if not file_path.exists():
        missing_files.append(file_name)

if missing_files:
    raise FileNotFoundError(f"Missing deployment files: {missing_files}")

print("All required deployment files are present.")