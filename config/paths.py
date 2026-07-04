from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'
INTERIM_DIR = BASE_DIR / 'data' / 'interim'
FEATURE_DIR = BASE_DIR / 'data' / 'features'
LABEL_DIR = BASE_DIR / 'data' / 'labels'
TRAINING_DIR = BASE_DIR / 'data' / 'training'
MODEL_STORE_DIR = BASE_DIR / 'model_store'
