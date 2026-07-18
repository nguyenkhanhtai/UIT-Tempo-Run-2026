import os
import time
import zipfile
import subprocess
import json

def create_rclone_config(script_dir):
    """Create a local rclone.conf file dynamically if token exists"""
    token_path = os.path.join(script_dir, 'rclone_token.json')
    conf_path = os.path.join(script_dir, 'local_rclone.conf')
    
    if os.path.exists(token_path):
        with open(token_path, 'r') as f:
            token_dict = json.load(f)
            token_string = json.dumps(token_dict)
            
        with open(conf_path, 'w') as f:
            f.write(f"[drive]\ntype = drive\ntoken = {token_string}\n")
        return conf_path
    return None

def cleanup_rclone_config(conf_path):
    """Remove the temporary local_rclone.conf file"""
    if conf_path and os.path.exists(conf_path):
        os.remove(conf_path)

def run_rclone_copyto(source, dest, conf_path=None):
    """Run rclone copyto to download a file and optionally rename it"""
    cmd = [
        "rclone", "copyto", source, dest,
        "--progress"
    ]
    if conf_path:
        cmd.extend(["--config", conf_path])
        
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0

def download_with_retry(source, dest, desc, conf_path=None, max_retries=5):
    """Download a file with retries"""
    print("\n" + "="*40)
    print(f"Downloading {desc} using rclone...")
    
    for attempt in range(1, max_retries + 1):
        print(f"Starting download with rclone (Attempt {attempt})...")
        if run_rclone_copyto(source, dest, conf_path):
            print(f"Successfully downloaded 100% of {desc}!")
            return True
        else:
            print(f"Download failed on attempt {attempt}.")
            if attempt < max_retries:
                print("Waiting 10 seconds before retrying...")
                time.sleep(10)
                
    print(f"Exceeded maximum retries for {desc}. Please check your rclone config or internet connection.")
    return False

def extract_and_cleanup_zip(zip_path, extract_dir, desc):
    """Extract a zip file and delete it to save space"""
    print("\n" + "="*40)
    if not os.path.exists(zip_path):
        print(f"{desc} not found. The download process might have failed.")
        return
        
    print(f"Extracting {desc}...")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction completed!")
    
    print(f"Deleting {zip_path} to save space...")
    try:
        os.remove(zip_path)
        print("Deleted successfully!")
    except Exception as e:
        print(f"Warning: Failed to delete the zip file: {e}")

def main():
    script_dir = os.path.dirname(__file__)
    dataset_dir = os.path.abspath(os.path.join(script_dir, '..', 'dataset'))
    artifacts_dir = os.path.abspath(os.path.join(script_dir, '..', 'artifacts'))
    
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    
    print(f"Dataset directory: {dataset_dir}")
    print(f"Artifacts directory: {artifacts_dir}")
    
    # 1. Setup Rclone Configuration
    conf_path = create_rclone_config(script_dir)
    
    try:
        # 2. Download Public round tasks.jsonl
        jsonl_source = "drive:AI Tempo Run/public_round_tasks.json"
        jsonl_dest = os.path.join(dataset_dir, "Public_round_tasks.jsonl")
        download_with_retry(jsonl_source, jsonl_dest, "Public round tasks.jsonl", conf_path)
        
        # 3. Download and Extract artifacts.zip
        artifacts_source = "drive:AI Tempo Run/artifacts.zip"
        root_dir = os.path.abspath(os.path.join(script_dir, '..'))
        artifacts_dest = os.path.join(root_dir, "artifacts.zip")
        if download_with_retry(artifacts_source, artifacts_dest, "artifacts.zip", conf_path):
            extract_and_cleanup_zip(artifacts_dest, root_dir, "artifacts.zip")

        # 4. Download and Extract keyframes.zip
        keyframes_source = "drive:AI Tempo Run/keyframes.zip"
        keyframes_dest = os.path.join(root_dir, "keyframes.zip")
        if download_with_retry(keyframes_source, keyframes_dest, "keyframes.zip", conf_path):
            extract_and_cleanup_zip(keyframes_dest, root_dir, "keyframes.zip")

        # 5. Download and Extract Video V3C.zip
        zip_source = "drive:AI Tempo Run/V3C.zip"
        zip_dest = os.path.join(dataset_dir, "Video_V3C.zip")
        if download_with_retry(zip_source, zip_dest, "Video V3C.zip", conf_path):
            extract_dir = os.path.join(dataset_dir, "Video_V3C")
            extract_and_cleanup_zip(zip_dest, extract_dir, "Video V3C.zip")
        

            
    finally:
        # 5. Cleanup local_rclone.conf regardless of success/failure
        cleanup_rclone_config(conf_path)
        
    print("\nScript completed successfully!")

if __name__ == "__main__":
    main()
