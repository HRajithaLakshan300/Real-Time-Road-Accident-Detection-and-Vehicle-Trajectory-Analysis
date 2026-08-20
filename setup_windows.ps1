$ErrorActionPreference = "Stop"

python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host "Environment created. Next: download the dataset into data/raw and run train_classifier.py."
