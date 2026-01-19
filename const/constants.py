from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parents[1]  
raw_data_path = os.path.join(ROOT_DIR, "data", "raw","people_segmentation")
process_data_path=os.path.join(ROOT_DIR, "data", "processed")
model_config_path=os.path.join(ROOT_DIR,"configs","model.yaml")
train_config_path=os.path.join(ROOT_DIR,"configs","train.yaml")
CHECKPOINT_PTH=os.path.join(ROOT_DIR,"checkpoints","best_model.h5")
ARTIFACTS_PATH=os.path.join(ROOT_DIR,"artifacts")
model_path=os.path.join(ROOT_DIR,"best_model.h5")