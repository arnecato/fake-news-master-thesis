import yaml
import subprocess

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def execute_commands(commands):
    for command in commands:
        cmd = command['cmd']
        print(f"Executing: {cmd}")
        subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    config = load_yaml('automation_pipeline.yaml')
    execute_commands(config['dataset'])