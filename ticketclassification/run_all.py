import subprocess
import sys
import os

# Remove unwanted paths
sys.path = [path for path in sys.path if 'set PYTHONPATH' not in path and 'CapstoneProjects' not in path]
print("Cleaned sys.path:", sys.path)

def run_script(script_path, description):
    print(f"\n[INFO] Starting: {description}")
    try:
        # Use the same Python interpreter that's running this script
        python_executable = sys.executable
        subprocess.run([python_executable, script_path], check=True)
        print(f"[INFO] Completed: {description}\n")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed: {description}. Error: {e}\n")
        exit(1)

def main():
    print("[INFO] Starting end-to-end process...")

    # Step 1: Data Preprocessing
    #run_script("src/data_processing.py", "Data Preprocessing")

    # Step 2: Model Fine-Tuning
    #run_script("src/train.py", "Model Fine-Tuning")

    # Step 3: Model Evaluation
    #run_script("src/evaluate.py", "Model Evaluation")

    # Step 4: Generate Predictions
    run_script("src/predict.py", "Generate Predictions")

    print("[INFO] End-to-end process completed successfully!")

if __name__ == "__main__":
    main()