import kagglehub
from pathlib import Path
import os
import shutil
ROOT_DIR = Path(__file__).resolve().parents[2]  # Current working directory
target_path = os.path.join(ROOT_DIR, "data", "raw")

# Create the directory if it doesn't exist
os.makedirs(target_path, exist_ok=True)

# Download to kagglehub's default cache location
cache_path = kagglehub.dataset_download('nikhilroxtomar/person-segmentation')

# Define target path


# Remove existing data if present
if os.path.exists(target_path):
    shutil.rmtree(target_path)

# Copy from cache to your desired location
shutil.copytree(cache_path, target_path)

print('Data source import complete.')
print('Downloaded from cache:', cache_path)
print('Copied to:', target_path)
print('ROOT_DIR:', ROOT_DIR)

# Free up space by removing the cache
print('\nRemoving cache folder to free up space...')
shutil.rmtree(cache_path)
print('Cache folder removed:', cache_path)

# Verify the data
print('\nContents of target directory:')
print(os.listdir(target_path))