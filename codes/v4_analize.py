import yaml, pickle, re
from argparse import Namespace
import pandas as pd, numpy as np

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
            t = re.sub("-", "", line.strip()).split(" ")
            epoch = t[t.index("at")+1]
            dataset = t[t.index("of")+1]
            continue
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
        elif state == "eval":
            res = [i for i in line.strip().split(":")[1].split(" ") if i != ""]
            res = pd.DataFrame(res).T
            res.columns = config['dataStats']['correctXPos']
            res.index = pd.Index([f"{epoch}_{dataset}"])
            evalRes = pd.concat([evalRes, res], axis = 0)

    others = list(set(trainRes.columns) - set(['JJR', 'WP','MD', "VBP"]))
    to = trainRes[others].replace("X", np.nan).map(float).mean(axis = 1, skipna=True).round(4)
    to.name = "OHTERS"
    eo = evalRes[others].replace("X", np.nan).map(float).mean(axis = 1, skipna=True).round(4)
    eo.name = "OTHERS"
    print("-------------------------------")
    print("train", "\n", pd.concat([trainRes[['JJR', 'WP','MD', "VBP"]], to], axis = 1))
    print("-------------------------------")
    print("eval", "\n", pd.concat([evalRes[['JJR', 'WP','MD', "VBP"]], eo], axis = 1))
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
    