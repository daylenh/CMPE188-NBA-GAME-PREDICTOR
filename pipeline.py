"""
NBA Predictor Pipeline Runner
Runs the entire ML pipeline automatically:
1. Data Collection
2. Preprocessing
3. Feature Engineering
4. Model Training
5. Streamlit App
"""

import subprocess
import sys
import os

def run_step(step_name, command):
    """Run a pipeline step and stop on failure"""
    print("\n" + "=" * 50)
    print(f"{step_name}")
    print("=" * 50)
    try:
        result = subprocess.run(command, check=True)
        if result.returncode == 0:
            print(f"{step_name} completed successfully!")
            return True
    except subprocess.CalledProcessError as e:
        print(f"{step_name} failed")
        print(e)
        sys.exit(1)

def main():
    print("NBA Predictor Automated Pipeline")
    print("===================================")
    # Get project root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_dir)
    # Fetch NBA Data
    run_step(
        "Step 1: Fetching NBA Data",
        [sys.executable, "data/get_data.py"]
    )
    # Preprocessing
    run_step(
        "Step 2: Preprocessing Data",
        [sys.executable, "src/preprocessing.py"]
    )
    # Feature Engineering
    run_step(
        "Step 3: Feature Engineering",
        [sys.executable, "src/feature_engineering.py"]
    )
    # Train Model
    run_step(
        "Step 4: Training Model",
        [sys.executable, "src/train_model.py"]
    )
    # Launch Streamlit
    print("\n" + "=" * 50)
    print("Launching Streamlit App")
    print("=" * 50)
    subprocess.run([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py"
    ])

if __name__ == "__main__":
    main()