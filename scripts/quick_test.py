import os
import sys
import subprocess

def main():
    print("🚀 Starting Quick Baseline Test for OPTIMIZED_TD3...")
    
    # Construct command
    # Assuming script is run from project root or scripts/ folder
    # We need to find train_single_agent.py
    
    project_root = os.getcwd()
    script_path = os.path.join(project_root, "train_single_agent.py")
    
    if not os.path.exists(script_path):
        # Try parent directory if run from scripts/
        project_root = os.path.dirname(project_root)
        script_path = os.path.join(project_root, "train_single_agent.py")
        
    if not os.path.exists(script_path):
        print(f"❌ Could not find train_single_agent.py in {os.getcwd()} or parent.")
        sys.exit(1)
        
    print(f"Found training script at: {script_path}")
    
    cmd = [
        sys.executable,
        script_path,
        "--algorithm", "OPTIMIZED_TD3",
        "--quick-test"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run command and stream output
        process = subprocess.Popen(
            cmd,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            encoding='utf-8',
            errors='replace'
        )
        
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        
        if process.returncode == 0:
            print("\n✅ Quick Test Completed Successfully!")
        else:
            print(f"\n❌ Quick Test Failed with return code {process.returncode}")
            
    except Exception as e:
        print(f"\n❌ Error running quick test: {e}")

if __name__ == "__main__":
    main()
