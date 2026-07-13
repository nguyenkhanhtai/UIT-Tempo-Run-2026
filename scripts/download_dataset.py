import os
import time
import zipfile
import shutil
import gdown

def main():
    # Define the path to save the dataset
    script_dir = os.path.dirname(__file__)
    dataset_dir = os.path.abspath(os.path.join(script_dir, '..', 'dataset'))
    cookies_path = os.path.join(script_dir, 'cookies.txt')
    
    # Create the directory if it does not exist
    os.makedirs(dataset_dir, exist_ok=True)
    print(f"Storage directory: {dataset_dir}")
    
    # 1. Download Public round tasks.jsonl
    print("\n" + "="*40)
    print("Downloading Public round tasks.jsonl...")
    jsonl_url = "https://drive.google.com/file/d/1sjslvmbv_jP9O83rKqB4TzJ3YWSMPzdM/view?usp=sharing"
    jsonl_output = os.path.join(dataset_dir, "Public_round_tasks.jsonl")
    gdown.download(url=jsonl_url, output=jsonl_output, quiet=False)
    
    # 2. Download Video V3C.zip
    print("\n" + "="*40)
    print("Downloading Video V3C.zip (will automatically resume if disconnected)...")
    
    # Copy cookies.txt to gdown cache directory for authentication
    gdown_cache_dir = os.path.expanduser("~/.cache/gdown")
    os.makedirs(gdown_cache_dir, exist_ok=True)
    if os.path.exists(cookies_path):
        print("Using cookies.txt for authentication...")
        shutil.copy(cookies_path, os.path.join(gdown_cache_dir, 'cookies.txt'))
    
    zip_id = "1dX1bCthvy_9Q_qRS5Qf_17zlsaziYs61"
    zip_output = os.path.join(dataset_dir, "Video_V3C.zip")
    
    max_retries = 1000 # Increased to ensure it resumes even if disconnected for a long time
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Starting download with gdown (Attempt {attempt})...")
            # resume=True enables resuming from the downloaded part instead of restarting
            result = gdown.download(id=zip_id, output=zip_output, quiet=False, resume=True, use_cookies=True)
            if result:
                print("Successfully downloaded 100% of Video V3C.zip!")
                break
        except Exception as e:
            print(f"Disconnected or encountered an error on attempt {attempt}: {e}")
            print("Waiting 10 seconds before retrying to resume...")
            time.sleep(10)
    else:
        print("Exceeded maximum retries. Please check your cookies configuration or internet connection.")
    
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
    
    print("\nScript completed successfully!")

if __name__ == "__main__":
    main()
