import yaml, pickle
from argparse import ArgumentParser, Namespace



def main(args, yaml_config):
    
    with open(yaml_config['originalData']['raw_CT'], "rb") as f:
        dataset_CT = pickle.load(f)

    pass


if __name__ == "__main__":
    
    args = Namespace(
        yaml_path = "/home/hyohyeongjang/2024SWELL/codes/v2_config.yaml"
    )
    
    with open(args.yaml_path, 'r') as file:
        yaml_config = yaml.safe_load(file)
    