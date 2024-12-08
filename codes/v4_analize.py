import yaml, pickle
from argparse import Namespace
import pandas as pd

def main(args, config):
    
    state = "None"
    dataset = "Baseline"
    epoch = "0"
    trainRes = pd.DataFrame(columns = config['dataStats']['correctXPos'])
    evalRes = pd.DataFrame()
    with open(config['analysis'], "r") as f:
        x = f.readlines()
    
    for line in x:
        if "eval" in line:
            state = "eval"
            continue
        elif "subEpoch" in line:
            t = line.split(":")[1].split(" ")
            dataset = t[t.index("subEpoch") + 1]
            epoch = t[t.index("Epoch")+1][0]
            state = "train"
            continue
        elif "train" in line:
            state = "train"
            continue
        else:
            pass
    
        if state == "train":
            res = [i for i in line.strip().split(":")[1].split(" ") if i != ""]
            res = pd.DataFrame(res).T
            res.columns = config['dataStats']['correctXPos']
            res.index = pd.Index([f"{dataset}_{epoch}"])
            trainRes = pd.concat([trainRes, res], axis = 0)
        elif state == "eval":
            res = [i for i in line.strip().split(":")[1].split(" ") if i != ""]
            res = pd.DataFrame(res).T
            res.columns = config['dataStats']['correctXPos']
            res.index = pd.Index([f"{dataset}_{epoch}"])
            evalRes = pd.concat([evalRes, res], axis = 0)
    
    print("train", "\n\n", trainRes[['JJR', "PRP", "VBP", "WDT"]])
    print("-----------")
    print("eval", "\n\n", evalRes[['JJR', "PRP", "VBP", "WDT"]])






if __name__ == "__main__":
    args = Namespace(
        # yaml_path = "/home/hyohyeongjang/2024SWELL/codes/v2_config.yaml",
        # yaml_path = "/home/hyohyeongjang/2024SWELL/codes/v3_config.yaml",
        yaml_path = "/home/hyohyeongjang/2024SWELL/codes/v4_config.yaml",
    )
    
    with open(args.yaml_path, 'r') as file:
        yaml_config = yaml.safe_load(file)
    
    #dictionary type string to dictionary
    yaml_config['dataStats']['labelToId'] = eval(yaml_config['dataStats']['labelToId'])
    yaml_config['dataStats']['labelToId'] = {"'"+"'" if key == "'" else key : value for key, value in yaml_config['dataStats']['labelToId'].items()}
    yaml_config['dataStats']['idToLabel'] = {value: key for key, value in yaml_config['dataStats']['labelToId'].items()}

    main(args, yaml_config)
    