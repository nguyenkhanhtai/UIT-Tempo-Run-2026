import os
import time
import argparse
import subprocess
import json

def create_rclone_config(script_dir):
    """Create a local rclone.conf file dynamically if token exists"""
    token_path = os.path.join(script_dir, '..', 'rclone_token.json')
    conf_path = os.path.join(script_dir, 'local_rclone.conf')
    
    if os.path.exists(token_path):
        with open(token_path, 'r') as f:
            token_dict = json.load(f)
            token_string = json.dumps(token_dict)
            
        with open(conf_path, 'w') as f:
            f.write(f"[drive]\ntype = drive\ntoken = {token_string}\n")
        return conf_path
    else:
        print(f"Warning: rclone_token.json not found at {token_path}")
    return None

def cleanup_rclone_config(conf_path):
    """Remove the temporary local_rclone.conf file"""
    if conf_path and os.path.exists(conf_path):
        os.remove(conf_path)

def run_rclone_upload(source, dest, conf_path=None):
    """Run rclone copy to upload a file or directory"""
    cmd = [
        "rclone", "copy", source, dest,
        "--progress",
        "--drive-chunk-size", "256M",
        "--tpslimit", "2"
    ]
    if conf_path:
        cmd.extend(["--config", conf_path])
        
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0

def upload_with_retry(source, dest, max_retries=5, conf_path=None):
    """Upload a file or directory with retries"""
    print("\n" + "="*40)
    print(f"Uploading '{source}' to '{dest}' using rclone...")
    
    for attempt in range(1, max_retries + 1):
        print(f"Starting upload with rclone (Attempt {attempt})...")
        if run_rclone_upload(source, dest, conf_path):
            print(f"Successfully uploaded 100% of '{source}'!")
            return True
        else:
            print(f"Upload failed on attempt {attempt}.")
            if attempt < max_retries:
                print("Waiting 10 seconds before retrying...")
                time.sleep(10)
                
    print(f"Exceeded maximum retries. Please check your internet connection or Google Drive space.")
    return False

def main():
    parser = argparse.ArgumentParser(description="Upload a file or directory to Google Drive 'AI Tempo Run' folder.")
    parser.add_argument("source", help="The source file or directory to upload")
    parser.add_argument("--dest", default="drive:AI Tempo Run", help="The destination folder on Google Drive (default: drive:AI Tempo Run)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_path = os.path.abspath(args.source)
    
    if not os.path.exists(source_path):
        print(f"Error: Source '{source_path}' does not exist.")
        return

    # 1. Setup Rclone Configuration
    conf_path = create_rclone_config(script_dir)
    
    try:
        # 2. Start Upload
        upload_with_retry(source_path, args.dest, conf_path=conf_path)
    finally:
        # 3. Cleanup local_rclone.conf
        cleanup_rclone_config(conf_path)
        
    print("\nScript completed successfully!")

if __name__ == "__main__":
    main()
