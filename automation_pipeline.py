import yaml
import subprocess
'''
automation_pipeline.yaml

model_training:
  - name: "ga_nsa"
  tool: "python .\ga_nsa.py"
  max_amount: 2000
  convergence_every: 100
  coverage: 0.001 # required increase in negative space coverage
  dims:
    - 2
    - 3
    - 4 
  embeddings:
    - bert
    - word2vec
  self_region:
    2: 0.03
    3: 0.01
    4: 0.005
  dataset_path: "dataset/ISOT/True_Fake_{embedding}_umap_{dim}dim_4000_4000_21417.h5"
  output_path: "model/detector/detectors_{embedding}_{dim}dim_4000_4000_21417.json"
'''
def model_training():
    with open('automation_pipeline.yaml', 'r') as file:
        config = yaml.safe_load(file)

    commands = []
    for model in config['model_training']:
        for dim in model['dims']:
            for embedding in model['embeddings']:
                dataset_path = model['dataset_path'].format(embedding=embedding, dim=dim)
                output_path = model['output_path'].format(embedding=embedding, dim=dim)
                self_region = model['self_region'][dim]
                command = (
                    f"{model['tool']} --dim={dim} --dataset={dataset_path} "
                    f"--detectorset={output_path} --amount={model['max_amount']} "
                    f"--convergence_every={model['convergence_every']} "
                    f"--self_region={self_region} --coverage={model['coverage']} "
                    f"--auto=1 --sample={model['sample_size']}"
                )
                commands.append(command)

    for cmd in commands:
        subprocess.run(cmd, shell=True)
def main():
    model_training()

if __name__ == "__main__":
    main()