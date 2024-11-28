from datetime import datetime
from collections import Counter
import sys, os, pandas as pd, numpy as np
from functools import reduce, partial
import pickle
from multiprocessing import Pool
from argparse import Namespace
import re
from knockknock import slack_sender
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
tqdm.pandas()

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import lmppl
import nltk
from nltk.corpus import wordnet

sys.path.append(os.path.expanduser('~/'))
from myUtils.timeUtils import TimeUtils
from myUtils.IOUtils import IOUtils
from myUtils.parallelUtils import ParallelUtils
import torch , logging, transformers


import random
import numpy as np
import torch

seed = 2021
deterministic = True

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if deterministic:
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False



device = "cuda" if torch.cuda.is_available() else "cpu"


def _get_includeData(tokenInfo, data): # token: str, data: List[str]
    return data.map(lambda x: tokenInfo in x)

def _select_ppl_samples(word, idx_score, ratios, data, max_len, max_sample_size):

    # idx_score = 나중에 수정하자 score distribution이 이상하니까 아래 0.05부분은 버리자 
    idx_score.sort_values(by = "diff", ascending = False) # descending, from 'helpful' when exists
    
    ppl_minus = idx_score[idx_score['diff'] < 0]
    ppl_plus = idx_score[idx_score['diff'] >= 0]
    
    exist = data[idx_score['index']]
    notExist = data[data.index.difference(idx_score['index'])].sample(frac = 1, random_state = 2021)
    print("lens", len(exist), len(notExist), len(data))

    notExist = notExist[notExist.map(lambda x : 100 < len(x.split(" ")))]

    exist_tkn = exist.map(lambda x: min(len(x.split(" ")), 512))
    notExist_tkn = notExist.map(lambda x: min(len(x.split(" ")), 512))    

    tot_tokens = sum(exist_tkn) + sum(notExist_tkn)
    max_tokens = int((max_sample_size / len(exist)) * sum(exist_tkn))
    
    print("------------------------------------------------------------")
    print("max_sample_size", max_sample_size, len(exist), len(notExist))

    # 상위 n개(엄밀하지는 않음) -> 대신 총합 ppl을 제시해서 비슷한 값 사이에서 비교할 수 있도록. 
    lst_exist, lst_nonExist, cum_ppl = [], [], []
    print("get ppl based samples")
    for ratio in tqdm(ratios):

        tkn_exist = int(max_tokens * ratio)
        tkn_notExist = int(max_tokens - tkn_exist)
        print("--", tkn_exist, tkn_notExist)

        t1 = 0
        c1 = -1
        for i, v in exist_tkn.items():
            if t1 < tkn_exist:
                c1 += 1
                t1 += v
            else:
                break

        # get exist_sents by ppl ranking
        if c1 == -1:
            exist_sents = pd.Series([]).reset_index(drop = True)
            exist_selected_ppl = 0
        else:
            exist_selected = idx_score.iloc[:c1]
            exist_sents = exist[exist_selected['index']].reset_index(drop = True)
            exist_selected_ppl = exist_selected['diff'].sum()

        t2 = 0
        c2 = -1
        for i, v in notExist_tkn.items():
            if t2 < tkn_notExist:
                c2 += 1
                t2 += v
            else:
                break

        if c2 == -1:
            notExist_sents = pd.Series([]).reset_index(drop = True)      
        else:
            notExist_sents = notExist.iloc[:c2].reset_index(drop = True)


        lst_exist.append(exist_sents)
        lst_nonExist.append(notExist_sents)

        cum_ppl.append(exist_selected_ppl)
    
    for i, j  in zip(lst_exist, lst_nonExist):
        print("token length", sum(i.map(len)), sum(j.map(len)), sum(i.map(len)) + sum(j.map(len)))
        print("sentence count", len(i), len(j), len(i)+len(j))

    return lst_exist, lst_nonExist, cum_ppl
        

# randomly sample. 문장의 정보값에 따라 적절하지 않을 수도 있음 -> select_ppl_samples
def _select_count_samples(word, ids, ratios, data, max_sample_size):

    exist = data[ids].sample(frac = 1, random_state = 2021)
    notExist = data[data.index.difference(ids)].sample(frac = 1, random_state = 2021)
    notExist = notExist[notExist.map(lambda x : 100 < len(x.split(" ")))]
    print("lens", len(exist), len(notExist), len(data))

    exist_tkn = exist.map(lambda x: min(len(x.split(" ")), 512))
    notExist_tkn = notExist.map(lambda x: min(len(x.split(" ")), 512))

    tot_tokens = sum(exist_tkn) + sum(notExist_tkn)
    max_tokens = int((max_sample_size / len(exist)) * sum(exist_tkn))
    print("------------------------------------------------------------")
    print("max_sample_size", max_sample_size, len(exist), len(notExist))

    # cum_ppl is not used
    lst_exist, lst_nonExist, cum_ppl = [], [], []
    print("get count based samples")
    for ratio in tqdm(ratios):
        tkn_exist = int(max_tokens * ratio)
        tkn_notExist = int(max_tokens - tkn_exist)
        print("--", tkn_exist, tkn_notExist)

        t1 = 0
        c1 = -1
        for i, v in exist_tkn.items():
            if t1 < tkn_exist:
                c1 += 1
                t1 += v
            else:
                break

        if c1 == -1:
            exist_sents = pd.Series([]).reset_index(drop = True)
        else:
            exist_sents = exist.iloc[:c1].reset_index(drop = True)

       

        # get exist_sents by ppl ranking

        t2 = 0
        c2 = -1
        for i, v in notExist_tkn.items():
            if t2 < tkn_notExist:
                c2 += 1
                t2 += v
            else:
                break

        if c2 == -1:
            notExist_sents = pd.Series([]).reset_index(drop = True)
        else:
            notExist_sents = notExist.iloc[:c2].reset_index(drop = True)
            


        lst_exist.append(exist_sents)
        lst_nonExist.append(notExist_sents)

        cum_ppl.append(None)
    
    for i, j  in zip(lst_exist, lst_nonExist):
        print("token length", sum(i.map(len)), sum(j.map(len)), sum(i.map(len)) + sum(j.map(len)))
        print("sentence count", len(i), len(j), len(i)+len(j))
        

    return lst_exist, lst_nonExist, cum_ppl
    

def make_files(data, data_splitted, wordStats, types, max_sample_size, args): # List[str]

    data = pd.Series(data)
    data_splitted = pd.Series([set(i) for i in data_splitted])
    wordStats = wordStats # sampled dataframe will come

    print("extracting included sentence per token...")
    with ParallelUtils() as parallelUtils:
        parallelUtils.change_function(partial(_get_includeData, data = data_splitted))
        # series[(series[bools])]
        data_filtered = parallelUtils.do_series(wordStats['word'], pre_assign = False, num_cores = 3)
        data_filtered.name = "sents" # series[series[boolean(indeces)]]
         # 왜 이렇게 되지?
    idx_filtered = data_filtered.map(lambda x: x[x == True].index)
    idx_filtered.name = 'index'

    for word, idxes, in tqdm(zip(wordStats['word'], idx_filtered)):
        print(f"{word} is being processing...")
        ids = pd.Series(list(idxes)) # where the word is in the sentences
        ids.name = "index"

        #cum_ppl in here is not used
        exist_count, notExist_count, cum_ppl_count = _select_count_samples(word, ids, args.ratio, data, max_sample_size)


        with open(args.scores_original_CT.format(word), "rb") as f:
            scores_original = pickle.load(f)
        with open(args.scores_masked_CT.format(word), "rb") as f:
            scores_masked = pickle.load(f)

        scores_diff = pd.Series((np.array(scores_masked) - np.array(scores_original)))
        scores_diff.name = "diff"
        idx_score = pd.concat([ids, scores_diff], axis = 1).sort_values(by = "diff")

        exist_ppl, notExist_ppl, cum_ppl_ppl = _select_ppl_samples(word, idx_score, args.ratio, data, len(scores_diff), max_sample_size)

        if types == "ft":
            dirs = args.data_FT
        if types == "ct":
            dirs = args.data_CT


        for exist, notExist, ppl, ratio in zip(exist_count, notExist_count, cum_ppl_count, args.ratio):
            with open(dirs.format(word, "count", ratio), "wb") as f:
                pickle.dump([exist, notExist, ppl], f)

        for exist, notExist, ppl, ratio in zip(exist_ppl, notExist_ppl, cum_ppl_ppl, args.ratio):
            with open(dirs.format(word, "ppl", ratio), "wb") as f:
                pickle.dump([exist, notExist, ppl], f)



webhook_url = "https://hooks.slack.com/services/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"
@slack_sender(webhook_url=webhook_url, channel="mine")
def main(args):

    # Load a large corpus dataset for pretraining (e.g., Wikipedia, OpenWebText)
    with open(args.raw_CT, "rb") as f:
        dataset_CT = pickle.load(f)

    temp = [i.lower() for i in dataset_CT['text']]
    temp1 = [i.split(" ") for i in temp] # List[List[str]]
    
    with open(args.file_wordStats_CT, "rb") as f:
        wordStats = pickle.load(f)
        
    # and랑 the는 안 됨. when은 masked score가 없음.
    token_list = ['and', 'one', 'the', 'for', 'new', 'time', 'they', 'was', 'has', 'that', 'who']

    wordStats_bools = wordStats.apply(lambda x: x['word'] in token_list, axis = 1)
    wordStats = wordStats[wordStats_bools]
    max_sample_size = int(len(temp) * wordStats['rat'].min())
    print(max_sample_size)

    make_files(temp, temp1, wordStats, "ct", max_sample_size, args)
    
    
if __name__ == "__main__":

    # already downloaded
    # nltk.download('wordnet')
    # nltk.download('averaged_perceptron_tagger')

    args = Namespace(
        dataset_CT = ("wikitext", "wikitext-103-v1"), # used in anywhere
        dataset_FT = "conll2003", # used in anywhere
        raw_CT = "/home/hyohyeongjang/2024SWELL/data_raw/ct_raw.pk", # used in anywhere
        raw_FT = "/home/hyohyeongjang/2024SWELL/data_raw/ft_raw.pk", # used in anywhere

        bag_of_word_CT = "/home/hyohyeongjang/2024SWELL/meta/BOW_CT.pk", # used in getWords
        bag_of_word_FT = "/home/hyohyeongjang/2024SWELL/meta/BOW_FT.pk", # used in getWords
        cnt_pos_CT = "/home/hyohyeongjang/2024SWELL/meta/cnt_pos_CT.pk", # used in getWords
        cnt_pos_FT = "/home/hyohyeongjang/2024SWELL/meta/cnt_pos_FT.pk", # used in getWords
        file_wordStats_CT = "/home/hyohyeongjang/2024SWELL/meta/word_ct.pk", # used in getWords
        file_wordStats_FT = "/home/hyohyeongjang/2024SWELL/meta/word_ft.pk", # used in getWords
        
        scores_original_CT = "/home/hyohyeongjang/2024SWELL/scores/score_CT_original_{}.pk", # used in getScores
        scores_masked_CT = "/home/hyohyeongjang/2024SWELL/scores/score_CT_mask_{}.pk", # used in getScores
        checkpoint_pplModel = "meta-llama/Meta-Llama-3-8B-Instruct", # used in getScores
        
        num_cores = 40, # making dataset
        # ratio = [0, 0.2, 0.4, 0.6, 0.8, 1.0], # used in getFiles
        ratio = [0, 0.3, 0.6, 1.0],
        data_CT = "/home/hyohyeongjang/2024SWELL/data_CT/CT_{}_{}_{}.pk", # used in getFiles
        data_FT = "/home/hyohyeongjang/2024SWELL/data_FT/FT_{}_{}_{}.pk", # used in getFiles

        checkpoint_baseModel = "FacebookAI/roberta-base", # used in continualTrain
        checkpoint_CTModel = "/home/hyohyeongjang/2024SWELL/weights/CT/CT_{}_{}_{}", # used in continualTrain
        checkpoint_FTModel = "/home/hyohyeongjang/2024SWELL/weights/FT/FT_{}_{}_{}_{}", # used in continualTrain
        max_seq_len = 512,
        batch_size = 64,
        do_RandomInitialize = False,
        num_cores_train = 10

    )
    
    # word = "example"  # Replace with the word you want to check
    pos_tagged = nltk.pos_tag(["a", "the"])
    print(pos_tagged)
    main(args)
