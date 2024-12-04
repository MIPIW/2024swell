import pickle
from argparse import Namespace
import numpy as np
import pandas as pd

if __name__ == "__main__":
    words = ['and', 'one', 'the', 'for', 'red', 'can', 'art', 'her', 'not', 'out', 'was', 'which']
    score_path = "/home/hyohyeongjang/2024SWELL/scores"
    # data_types = ['CT'] only CT
    ppl_types = ["mask", "original"]

    dic = {i: [] for i in words}
    for k in ppl_types:
        for j in words:
            s = f"{score_path}/score_CT_{k}_{j}.pk"
            dic[j].append(s)

    for key in dic.keys():

        mask = dic[key][0]
        original = dic[key][1]
        
        with open(mask, "rb") as f:
            m = pickle.load(f)

        with open(original, "rb") as f:
            o = pickle.load(f)

        print(key, "\n", (pd.Series(m) - pd.Series(o)).quantile([0.05, 0.995]), end = "\n\n")

                

