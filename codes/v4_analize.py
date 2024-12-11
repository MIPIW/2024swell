import yaml, pickle, re
from argparse import Namespace
import pandas as pd, numpy as np

def get_args(line, state):

    t = re.sub("-", "", line.strip()).split(" ")
    epoch = t[t.index("at")+1]

    return epoch, state

def get_dfs(line, config, epoch):
    res = [i for i in line.strip().split(":")[1].split(" ") if i != ""]
    res = pd.DataFrame(res).T
    res.columns = config['dataStats']['correctXPos']
    res.index = pd.Index([f"{epoch}"])

    return res

def get_others(res, target):
    others = list(set(res.columns) - set(target))
    to = res[others].replace("X", np.nan).map(float).mean(axis = 1, skipna=True).round(4)
    to.name = "OHTERS"
    return to 

def main(args, config):
    target = ['NN', 'IN']
    state = "None"
    epoch = "0"

    trainRes = pd.DataFrame(columns = config['dataStats']['correctXPos'])
    evalResId = pd.DataFrame(columns = config['dataStats']['correctXPos'])
    evalResOOD = pd.DataFrame(columns = config['dataStats']['correctXPos'])
    evalResTd = pd.DataFrame(columns = config['dataStats']['correctXPos'])
    
    with open(config['analysis'], "r") as f:
        x = f.readlines()
    
    for line in x:
        if "eval" in line:
            epoch, state = get_args(line, "eval")
            continue
        elif "train" in line:
            epoch, state = get_args(line, "train")
            
            continue
        else:
            pass

        res = get_dfs(line, config, epoch)
        if state == "train":
            trainRes = pd.concat([trainRes, res], axis = 0)
        elif state == "eval":
            evalResId = pd.concat([evalResId, res], axis = 0)

    to = get_others(trainRes, target)
    eo = get_others(evalResId, target)
    
    print("-------------------------------")
    print("train", "\n", pd.concat([trainRes[target], to], axis = 1))
    print("-------------------------------")
    print("eval id", "\n", pd.concat([evalResId[target], eo], axis = 1))
    print("-------------------------------")






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

    main(args, yaml_config)
    