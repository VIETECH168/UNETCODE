import os
import requests
import tarfile
import zipfile

def download_file(url, destination):
    response = requests.get(url)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")

def extract_file(filepath, extract_to):
    if filepath.endswith('.tar.gz') or filepath.endswith('.tgz'):
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(path=extract_to)
    elif filepath.endswith('.zip'):
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    else:
        raise ValueError("Unsupported file format")

# Define paths
dataset_url = 'https://github.com/alexgkendall/SegNet-Tutorial/releases/download/v1.0/camvid.tar.gz'
data_dir = 'camvid'
compressed_file = os.path.join(data_dir, 'camvid.tar.gz')

# Create directory if it does not exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Download dataset
print("Downloading dataset...")
download_file(dataset_url, compressed_file)

# Extract dataset
print("Extracting dataset...")
extract_file(compressed_file, data_dir)

print(f"Dataset extracted to {data_dir}")
