# Download the Kaggle dataset

## Option 1: Kaggle website

1. Open the dataset page.
2. Select **Download**.
3. Extract the archive into `data/raw/`.

The final structure should be similar to:

```text
data/raw/
├── train/
│   ├── Accident/
│   └── Non Accident/
├── val/
│   ├── Accident/
│   └── Non Accident/
└── test/
    ├── Accident/
    └── Non Accident/
```

The training script searches nested folders, so one extra extracted parent folder is acceptable.

## Option 2: Kaggle CLI

Install and authenticate the Kaggle command-line tool, then run from the project root:

```powershell
kaggle datasets download -d ckay16/accident-detection-from-cctv-footage -p data/raw --unzip
```

Inspect the result:

```powershell
python scripts/inspect_dataset.py --data-dir data/raw
```
