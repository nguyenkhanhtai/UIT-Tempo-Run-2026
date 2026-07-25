import sys
import os
import json
import subprocess
import shutil

def main():
    args = sys.argv[1:]
    
    tasks_file = None
    out_file = None
    
    for i, arg in enumerate(args):
        if arg == "--tasks":
            tasks_file = args[i+1]
        elif arg == "--out":
            out_file = args[i+1]
            
    if not tasks_file:
        print("Missing --tasks argument")
        sys.exit(1)
        
    if not out_file:
        out_file = "submission.json"
        
    # Determine final output directory early
    import glob
    if out_file == "submission.json":
        os.makedirs("submission", exist_ok=True)
        sub_dirs = glob.glob("submission/*")
        valid_dirs = [d for d in sub_dirs if os.path.basename(d).isdigit()]
        if not valid_dirs:
            next_id = 1
        else:
            valid_dirs.sort(key=lambda x: int(os.path.basename(x)))
            next_id = int(os.path.basename(valid_dirs[-1])) + 1
        final_out_dir = os.path.join("submission", f"{next_id:03d}")
        os.makedirs(final_out_dir, exist_ok=True)
        final_out_file = os.path.join(final_out_dir, "submission.json")
    else:
        final_out_dir = os.path.dirname(os.path.abspath(out_file))
        if not final_out_dir:
            final_out_dir = "."
        os.makedirs(final_out_dir, exist_ok=True)
        final_out_file = out_file
        
    with open(tasks_file, 'r') as f:
        lines = f.readlines()
        
    num_chunks = int(os.environ.get("NUM_CHUNKS", "3"))
    chunk_size = (len(lines) + num_chunks - 1) // num_chunks
    
    print(f"Splitting {len(lines)} tasks into {num_chunks} chunks of size {chunk_size}...")
    
    chunk_files = []
    for i in range(num_chunks):
        chunk_lines = lines[i*chunk_size : (i+1)*chunk_size]
        if not chunk_lines:
            break
        chunk_file = f"{tasks_file}.chunk{i}.jsonl"
        with open(chunk_file, 'w') as f:
            f.writelines(chunk_lines)
        chunk_files.append(chunk_file)
        
    chunk_dirs = []
    
    for i, chunk_file in enumerate(chunk_files):
        print(f"\n==========================================================")
        print(f"Running Chunk {i+1}/{len(chunk_files)}: {chunk_file}")
        print(f"==========================================================\n")
        
        chunk_out_dir = f"submission_chunk_{i}"
        chunk_out_file = os.path.join(chunk_out_dir, "submission.json")
        
        if os.path.exists(chunk_out_file):
            print(f"Found existing chunk results in {chunk_out_dir}. Reusing it!")
            chunk_dirs.append(chunk_out_dir)
            continue
        
        new_args = []
        skip_next = False
        for arg in args:
            if skip_next:
                skip_next = False
                continue
            if arg == "--tasks":
                new_args.extend(["--tasks", chunk_file])
                skip_next = True
            elif arg == "--out":
                new_args.extend(["--out", chunk_out_file])
                skip_next = True
            else:
                new_args.append(arg)
                
        # If --out wasn't in original args
        if "--out" not in args:
            new_args.extend(["--out", chunk_out_file])
                
        cmd = ["uv", "run", "python", "pipeline/retrieve.py"] + new_args
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        
        res = subprocess.run(cmd, env=env)
        if res.returncode != 0:
            print(f"Chunk {i+1} failed with exit code {res.returncode}")
            sys.exit(res.returncode)
            
        chunk_dirs.append(chunk_out_dir)
        
    print("\n==========================================================")
    print("Merging results...")
    print("==========================================================\n")
    
    all_clean_preds = []
    all_detailed_preds = []
    config_data = {}
    
    for cdir in chunk_dirs:
        with open(os.path.join(cdir, "submission.json"), "r") as f:
            sub = json.load(f)
            all_clean_preds.extend(sub.get("predictions", []))
            
        with open(os.path.join(cdir, "detailed_submission.json"), "r") as f:
            d_sub = json.load(f)
            all_detailed_preds.extend(d_sub.get("predictions", []))
            
        config_path = os.path.join(cdir, "config.json")
        if os.path.exists(config_path) and not config_data:
            with open(config_path, "r") as f:
                config_data = json.load(f)
                
    # Write merged JSON files
    with open(final_out_file, "w") as f:
        json.dump({"predictions": all_clean_preds}, f)
        
    detailed_out_file = os.path.join(final_out_dir, "detailed_submission.json")
    with open(detailed_out_file, "w") as f:
        json.dump({"predictions": all_detailed_preds}, f, indent=2)
        
    if config_data:
        with open(os.path.join(final_out_dir, "config.json"), "w") as f:
            json.dump(config_data, f, indent=2)
            
    # Zip the final submission
    import zipfile
    zip_path = os.path.join(final_out_dir, "submission.zip")
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.write(final_out_file, arcname="submission.json")
        
    # Merge figures
    final_fig_dir = os.path.join(final_out_dir, "figures", "task")
    os.makedirs(final_fig_dir, exist_ok=True)
    
    for cdir in chunk_dirs:
        chunk_fig_dir = os.path.join(cdir, "figures", "task")
        if os.path.exists(chunk_fig_dir):
            for task_folder in os.listdir(chunk_fig_dir):
                src = os.path.join(chunk_fig_dir, task_folder)
                dst = os.path.join(final_fig_dir, task_folder)
                if os.path.exists(src):
                    shutil.move(src, dst)
                    
    # Clean up temp chunks
    for cfile in chunk_files:
        if os.path.exists(cfile):
            os.remove(cfile)
    for cdir in chunk_dirs:
        if os.path.exists(cdir):
            shutil.rmtree(cdir)
            
    # Clean up any remaining chunks in the root directory
    for i in range(num_chunks):
        old_dir = f"submission_chunk_{i}"
        if os.path.exists(old_dir):
            shutil.rmtree(old_dir)
            
    print(f"[done] wrote merged outputs to {final_out_dir}")

if __name__ == "__main__":
    main()
