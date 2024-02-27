import subprocess
import yaml
def main():

    with open("parameters.yaml", 'r') as stream:
        parameters = yaml.safe_load(stream)
    if parameters['copy_mode']==False:
        command = f"torchrun --nnodes {parameters['nnodes']} --nproc_per_node {parameters['nproc_per_node']} run_new_tech_train_step.py"
        subprocess.run(command, shell=True)
if __name__ == "__main__":
    main()
