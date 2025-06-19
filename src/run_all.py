import pathlib
import subprocess
RUN_DIR = "FracGradient/src/"

def main():
    # run all main*.py files in the RUN_DIR
    run_dir = pathlib.Path(RUN_DIR)
    main_files = list(run_dir.glob("main*.py"))
    for main_file in main_files:
        print(f"Running {main_file.name}...")
        # run python3 RUN_DIR + main_file.name
        subprocess.run(["python3", str(main_file)], check=True)
    print("All main files executed successfully.")

if __name__ == "__main__":
    main()