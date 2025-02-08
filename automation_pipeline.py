import yaml
import subprocess
from datetime import datetime

def document_embedding():
    with open('automation_pipeline.yaml', 'r') as file:
        config = yaml.safe_load(file)

    commands = []
    doc_embed_config = config['document_embedding']
    for model in doc_embed_config['models']:
        command = f"{model}"
        commands.append(command)

    for cmd in commands:
        print(cmd)
        subprocess.run(cmd, shell=True)
        

def umap_dimensionality_reduction():
    with open('automation_pipeline.yaml', 'r') as file:
        config = yaml.safe_load(file)
    commands = []
    umap_config = config['umap_dimensionality_reduction']
    for dim in umap_config['dims']:
        for embedding in umap_config['embeddings']:
            command = umap_config['tool'].format(embedding=embedding, dim=dim)
            commands.append(command)

    for cmd in commands:
        print(f"{datetime.now()}: {cmd}")    
        subprocess.run(cmd, shell=True)
         
def model_training(num_experiments):
    with open('automation_pipeline.yaml', 'r') as file:
        config = yaml.safe_load(file)

    commands = []
    for model in config['model_training']:
        for dim in model['dims']:
            for embedding in model['embeddings']:
                for i in range(num_experiments):
                  dataset_path = model['dataset_path'].format(embedding=embedding, dim=dim)
                  output_path = model['output_path'].format(embedding=embedding, dim=dim)
                  self_region = model['self_region'][dim]
                  command = (
                      f"{model['tool']} --dim={dim} --dataset={dataset_path} "
                      f"--detectorset={output_path} --amount={model['max_amount']} "
                      f"--convergence_every={model['convergence_every']} "
                      f"--self_region={self_region} --coverage={model['coverage']} "
                      f"--auto=1 --sample={model['sample_size']} --experiment={i}"
                  )
                  commands.append(command)

    for cmd in commands:
        print(cmd)
        subprocess.run(cmd, shell=True)

def main():
    #document_embedding()
    #umap_dimensionality_reduction()    
    model_training(1)

if __name__ == "__main__":
    main()