import yaml
import subprocess

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def execute_commands(dataset):
    for command in dataset['cmd']:
        print(f"Executing: {command}")
        subprocess.run(command, shell=True, check=True)

if __name__ == "__main__":
    config = load_yaml('automation_pipeline.yaml')
    execute_commands(config['dataset'])