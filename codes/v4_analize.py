import yaml, pickle, re
from argparse import Namespace
import pandas as pd, numpy as np

def get_args(line, state):

    t = re.sub("-", "", line.strip()).split(" ")
    epoch = t[t.index("at")+1]
    dataset = t[t.index("of")+1]

    return epoch, dataset, state

def main(args, config):
    target = ['JJR', 'WDT']
    state = "None"
    dataset = "Baseline"
    epoch = "0"
    trainRes = pd.DataFrame(columns = config['dataStats']['correctXPos'])
    evalResId = pd.DataFrame(columns = config['dataStats']['correctXPos'])
    evalResOOD = pd.DataFrame(columns = config['dataStats']['correctXPos'])

    with open(config['analysis2'], "r") as f:
        x = f.readlines()
    
    for line in x:
        if "eval" in line:
            if "id" in line:
                epoch, dataset, state = get_args(line, "eval_id")
                continue
            if "ood" in line:
                epoch, dataset, state = get_args(line, "eval_ood")
                
                continue
            if "td" in line:
                state = "eval_td"
                t = re.sub("-", "", line.strip()).split(" ")
                epoch = t[t.index("at")+1]
                dataset = t[t.index("of")+1]

        elif "train" in line:
            t = re.sub("-", "", line.strip()).split(" ")
            epoch = t[t.index("at")+1]
            dataset = t[t.index("of")+1]
            state = "train"
            continue
        else:
            pass
    
        if state == "train":
            res = [i for i in line.strip().split(":")[1].split(" ") if i != ""]
            res = pd.DataFrame(res).T
            res.columns = config['dataStats']['correctXPos']
            res.index = pd.Index([f"{epoch}_{dataset}"])
            trainRes = pd.concat([trainRes, res], axis = 0)
        elif state == "eval_id":
            res = [i for i in line.strip().split(":")[1].split(" ") if i != ""]
            res = pd.DataFrame(res).T
            res.columns = config['dataStats']['correctXPos']
            res.index = pd.Index([f"{epoch}_{dataset}"])
            evalResId = pd.concat([evalResId, res], axis = 0)
        elif state == "eval_ood":
            res = [i for i in line.strip().split(":")[1].split(" ") if i != ""]
            res = pd.DataFrame(res).T
            res.columns = config['dataStats']['correctXPos']
            res.index = pd.Index([f"{epoch}_{dataset}"])
            evalResOOD = pd.concat([evalResOOD, res], axis = 0)


    others = list(set(trainRes.columns) - set(target))
    to = trainRes[others].replace("X", np.nan).map(float).mean(axis = 1, skipna=True).round(4)
    to.name = "OHTERS"
    eo = evalResId[others].replace("X", np.nan).map(float).mean(axis = 1, skipna=True).round(4)
    eo.name = "OTHERS"
    eoo = evalResOOD[others].replace("X", np.nan).map(float).mean(axis = 1, skipna=True).round(4)
    eoo.name = "OTHERS"

    print("-------------------------------")
    print("train", "\n", pd.concat([trainRes[target], to], axis = 1))
    print("-------------------------------")
    print("eval id", "\n", pd.concat([evalResId[target], eo], axis = 1))
    print("-------------------------------")
    print("eval ood", "\n", pd.concat([evalResOOD[target], eoo], axis = 1))
    print("-------------------------------")






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
    