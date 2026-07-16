import os
import time
import zipfile
import shutil
import subprocess

import json

def run_rclone_copyto(source, dest):
    """Run rclone copyto to download a file and optionally rename it"""
    cmd = [
        "rclone", "copyto", source, dest,
        "--progress", "--drive-shared-with-me"
    ]
    
    # Create a local rclone.conf file dynamically
    script_dir = os.path.dirname(__file__)
    token_path = os.path.join(script_dir, 'rclone_token.json')
    conf_path = os.path.join(script_dir, 'local_rclone.conf')
    
    if os.path.exists(token_path):
        with open(token_path, 'r') as f:
            token_json = f.read().strip()
            
        with open(conf_path, 'w') as f:
            f.write(f"[drive]\ntype = drive\ntoken = {token_json}\n")
            
        # Add --config flag to use the generated local conf
        cmd.extend(["--config", conf_path])
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0

def main():
    # Define the path to save the dataset
    script_dir = os.path.dirname(__file__)
    dataset_dir = os.path.abspath(os.path.join(script_dir, '..', 'dataset'))
    
    # Create the directory if it does not exist
    os.makedirs(dataset_dir, exist_ok=True)
    print(f"Storage directory: {dataset_dir}")
    
    # 1. Download Public round tasks.jsonl
    print("\n" + "="*40)
    print("Downloading Public round tasks.jsonl using rclone...")
    # NOTE: Thay đổi 'drive:' thành tên remote của bạn nếu khác
    jsonl_source = "drive:AI Tempo Run/public_round_tasks.json"
    jsonl_dest = os.path.join(dataset_dir, "Public_round_tasks.jsonl")
    if not run_rclone_copyto(jsonl_source, jsonl_dest):
        print("Failed to download Public_round_tasks.jsonl")
    else:
        print("Successfully downloaded Public_round_tasks.jsonl!")
    
    # 2. Download Video V3C.zip
    print("\n" + "="*40)
    print("Downloading Video V3C.zip using rclone (supports automatic resume)...")
    
    zip_source = "drive:AI Tempo Run/V3C.zip"
    zip_output = os.path.join(dataset_dir, "Video_V3C.zip")
    
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        print(f"Starting download with rclone (Attempt {attempt})...")
        if run_rclone_copyto(zip_source, zip_output):
            print("Successfully downloaded 100% of Video V3C.zip!")
            break
        else:
            print(f"Download failed on attempt {attempt}.")
            if attempt < max_retries:
                print("Waiting 10 seconds before retrying...")
                time.sleep(10)
    else:
        print("Exceeded maximum retries. Please check your rclone config or internet connection.")
        return
    
    # 3. Extract the zip file
    print("\n" + "="*40)
    if os.path.exists(zip_output):
        print("Extracting Video V3C.zip...")
        extract_dir = os.path.join(dataset_dir, "Video_V3C")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_output, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction completed!")
        
        # Delete zip file to save space
        print(f"Deleting {zip_output} to save space...")
        try:
            os.remove(zip_output)
            print("Deleted successfully!")
        except Exception as e:
            print(f"Warning: Failed to delete the zip file: {e}")
    else:
        print("Video V3C.zip not found. The download process might have failed.")
    
    # 4. Download artifacts.zip
    print("\n" + "="*40)
    print("Downloading artifacts.zip using rclone (supports automatic resume)...")
    
    artifacts_source = "drive:AI Tempo Run/artifacts.zip"
    artifacts_output = os.path.join(dataset_dir, "artifacts.zip")
    
    for attempt in range(1, max_retries + 1):
        print(f"Starting download with rclone (Attempt {attempt})...")
        if run_rclone_copyto(artifacts_source, artifacts_output):
            print("Successfully downloaded 100% of artifacts.zip!")
            break
        else:
            print(f"Download failed on attempt {attempt}.")
            if attempt < max_retries:
                print("Waiting 10 seconds before retrying...")
                time.sleep(10)
    else:
        print("Exceeded maximum retries for artifacts.zip. Please check your rclone config or internet connection.")
        return

    # 5. Extract artifacts.zip
    print("\n" + "="*40)
    if os.path.exists(artifacts_output):
        print("Extracting artifacts.zip...")
        artifacts_extract_dir = os.path.join(dataset_dir, "artifacts")
        os.makedirs(artifacts_extract_dir, exist_ok=True)
        with zipfile.ZipFile(artifacts_output, 'r') as zip_ref:
            zip_ref.extractall(artifacts_extract_dir)
        print("Extraction completed!")
        
        # Delete zip file to save space
        print(f"Deleting {artifacts_output} to save space...")
        try:
            os.remove(artifacts_output)
            print("Deleted successfully!")
        except Exception as e:
            print(f"Warning: Failed to delete the zip file: {e}")
    else:
        print("artifacts.zip not found. The download process might have failed.")
    
    # Cleanup local_rclone.conf
    conf_path = os.path.join(script_dir, 'local_rclone.conf')
    if os.path.exists(conf_path):
        os.remove(conf_path)
        
    print("\nScript completed successfully!")

if __name__ == "__main__":
    main()
