import yaml, pickle, re
from argparse import Namespace
import pandas as pd, numpy as np

def _get_args(line, state):

    t = re.sub("-", "", line.strip()).split(" ")
    epoch = t[t.index("at")+1]
    subEpoch = t[t.index("on")+1]

    return epoch, subEpoch, state

def _get_dfs(line, config, epoch, subepoch):
    res = [i for i in line.strip().split(":")[1].split(" ") if i != ""]
    res = pd.DataFrame(res).T
    res.columns = config['dataStats']['correctXPos']
    res.index = pd.Index([f"{epoch}_{subepoch}"])

    return res

def _get_others(res, target, open_close_dic):
    others = list(set(res.columns) - set(target))
    to = res[others].replace("X", np.nan).applymap(float) 
    close = [key for key, value in open_close_dic.items() if value == "CLOSE" and key != "IN"]
    open = [key for key, value in open_close_dic.items() if value == "OPEN" and key != "NN"]
    print(close)
    toClose = to[close].mean(axis = 1, skipna=True).round(4)
    toOpen = to[open].mean(axis = 1, skipna=True).round(4)
    to = pd.concat([toClose, toOpen], axis = 1)
    # to_close = /
    to.columns = ["CLOSE", 'OPEN']
    return to 

def get_result_dataframe(args, config):
    target = ['NN', 'IN']
    state = "None"
    epoch = "0"
    subepoch = "None"
    open_close_dic = yaml_config['analysis_pos']

    trainRes = pd.DataFrame(columns = config['dataStats']['correctXPos'])
    evalResId = pd.DataFrame(columns = config['dataStats']['correctXPos'])
    evalResOOD = pd.DataFrame(columns = config['dataStats']['correctXPos'])
    evalResTd = pd.DataFrame(columns = config['dataStats']['correctXPos'])
    
    fName = config['analysis']
    with open(fName, "r") as f:
        x = f.readlines()
    
    for line in x:
        if "eval" in line:
            epoch, subepoch, state = _get_args(line, "eval")
            continue
        elif "train" in line:
            epoch, subepoch, state = _get_args(line, "train")
            
            continue
        else:
            pass

        res = _get_dfs(line, config, epoch, subepoch)
        if state == "train":
            trainRes = pd.concat([trainRes, res], axis = 0)
        elif state == "eval":
            evalResId = pd.concat([evalResId, res], axis = 0)

    # print(trainRes)
    to = _get_others(trainRes, target, open_close_dic)
    eo = _get_others(evalResId, target, open_close_dic)
    
    x = pd.concat([trainRes[target], to], axis = 1)
    y = pd.concat([evalResId[target], eo], axis = 1)
    
    x.to_csv(fName.split(".")[0]+"_train.csv")
    y.to_csv(fName.split(".")[0]+"_eval.csv")

    return x, y

def token_frequency(args, config):
    
    with open(config['contFiles']['data_CT'].format("groups"), "rb") as f:
        grouped_df = pickle.load(f)
    
    print(grouped_df)

def main(args, config):
    # get_result_dataframe(args, config)

    token_frequency(args, config)


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
    