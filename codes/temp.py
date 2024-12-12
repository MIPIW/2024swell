from argparse import ArgumentParser, Namespace
import yaml
import pandas as pd

def main(args, config):

    train_log = config['analysis'].split(".")[0]+"_train.csv"
    eval_log = config['analysis'].split(".")[0]+"_eval.csv"

    train = pd.read_csv(train_log)
    eval = pd.read_csv(eval_log)

    print(train)




if __name__ == "__main__":
    args = Namespace(
        # yaml_path = "/home/hyohyeongjang/2024SWELL/codes/v2_config.yaml",
        # yaml_path = "/home/hyohyeongjang/2024SWELL/codes/v3_config.yaml",
        yaml_path = "/home/hyohyeongjang/2024SWELL/codes/v5_config.yaml",
    )
    
    with open(args.yaml_path, 'r') as file:
        yaml_config = yaml.safe_load(file)
    
    #dictionary type string to dictionary
    yaml_config['dataStats']['labelToId'] = eval(yaml_config['dataStats']['labelToId'])
    yaml_config['dataStats']['labelToId'] = {"'"+"'" if key == "'" else key : value for key, value in yaml_config['dataStats']['labelToId'].items()}
    yaml_config['dataStats']['idToLabel'] = {value: key for key, value in yaml_config['dataStats']['labelToId'].items()}
    yaml_config['analysis_pos'] = {"'"+"'" if key == "'" else key : value for key, value in yaml_config['analysis_pos'].items()}
    
    main(args, yaml_config)
    