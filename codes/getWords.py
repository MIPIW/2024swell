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



def get_dataAvgLength(data): # List[List[str]]
    x = [len(i) for i in data]
    return sum(x) / len(x)

def _filter_data(datum): # str
    if datum == None:
        return {}
    x = set(datum)
    x = set(filter(lambda x: 2 < len(x) < 10, x))

    return x    

def _is_included(token, data, total_length): #token: str(tokens)(splitted by pools)
    return sum([token in datum for datum in data]) / total_length

def get_dataStats(data, data_splitted, types, args): # List[str]

    print("get_dataStats starts...")
    if types == "ct" and os.path.isfile(args.file_wordStats_CT):
        print("final file exists...")
        with open(args.file_wordStats_CT, "rb") as f:
            out = pickle.load(f)
            return out    
        
    if types == "ft" and os.path.isfile(args.file_wordStats_FT):
        print("final file exists...")
        with open(args.file_wordStats_FT, "rb") as f:
            out = pickle.load(f)
            return out

    skip_BOW = False
    if types == "ct" and  os.path.isfile(args.bag_of_word_CT):
        print("bag of words exists...")
        with open(args.bag_of_word_CT, "rb") as f:
            data_filtered = pickle.load(f)
        skip_BOW = True

    if types == "ft" and  os.path.isfile(args.bag_of_word_FT):
        print("bag of words exists...")
        with open(args.bag_of_word_FT, "rb") as f:
            data_filtered = pickle.load(f)
        skip_BOW = True
    
    if not skip_BOW:
        print("get token individual frequency...")
        with ParallelUtils() as parallelUtils:
            parallelUtils.change_function(_filter_data)
            # get unigram counts    
            data_filtered = pd.Series(data_splitted)
            data_filtered = parallelUtils.do_series(data_filtered, pre_assign = False, num_cores = args.num_cores)
            data_filtered = data_filtered.to_list() # series[set[str(tokens)]]
            data_filtered = list(reduce(lambda x, y : x | y, data_filtered)) # list[str(tokens)]
    
        with open(args.bag_of_word_CT, "wb") as f:
            pickle.dump(data_filtered, f)


    skip_cnt = False
    if types == "ct" and os.path.isfile(args.cnt_pos_CT):
        print("cnt_pos exists...")
        with open(args.cnt_pos_CT, "rb") as f:
            data_cnt, data_pos = pickle.load(f)
            skip_cnt = True
    
    if not skip_cnt:
        print("get token count in each sentences per tokens...")

        data_splitted_set = [set(i) for i in data_splitted]
        f = partial(_is_included, data = data_splitted_set, total_length = len(data_splitted))
        data_cnt = process_map(f, data_filtered, max_workers = args.num_cores, chunksize = 1000)
        
        chunk_size = len(data_filtered) // args.num_cores
        total_length = len(data)
        chunks = [data_filtered[i * chunk_size:(i + 1) * chunk_size] for i in range(args.num_cores - 1)]
        chunks.append(data_filtered[(args.num_cores - 1) * chunk_size:])

        def split_into_sub_chunks(data, num_sub_chunks=100):
            sub_chunk_size = max(1, len(data) // num_sub_chunks)
            return [data[i * sub_chunk_size:(i + 1) * sub_chunk_size] for i in range(num_sub_chunks - 1)] + [data[(num_sub_chunks - 1) * sub_chunk_size:]]

        # Further divide each chunk into 100 smaller sub-chunks
        chunks = [sub_chunk for chunk in chunks for sub_chunk in split_into_sub_chunks(chunk, num_sub_chunks=100)]

        print("get token pos...")

        data_pos = process_map(nltk.pos_tag, chunks, max_workers = args.num_cores)
        data_pos = [i for j in data_pos for i in j] # unlist

        print(len(data_pos), len(data_cnt), len(data_filtered))

        with open(args.cnt_pos_CT, "wb") as f:
            pickle.dump([data_cnt, data_pos], f)

    
    print("postprocessing...")
    data_cnt_valid = list(filter(lambda x: 0.01 < x[1] < 0.9, enumerate(data_cnt)))
    data_tok = [data_pos[i[0]][0] for i in data_cnt_valid] # tokens
    data_pos = [data_pos[i[0]][1] for i in data_cnt_valid] # pos
    data_cnt = [data_cnt[i[0]] for i in data_cnt_valid] # proportion

    out = pd.DataFrame({i: (j, k) for i, j, k in zip(data_tok, data_pos, data_cnt)}).T
    out = out.reset_index(drop = False)
    out.columns = ["word", 'pos', 'rat']

    if types == "ct":
        with open(args.file_wordStats_CT, "wb") as f:
            pickle.dump(out, f)
    if types == "ft":
        with open(args.file_wordStats_FT, "wb") as f:
            pickle.dump(out, f)
    
    return out




webhook_url = "https://hooks.slack.com/services/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"
@slack_sender(webhook_url=webhook_url, channel="mine")
def main(args):

    # Load a large corpus dataset for pretraining (e.g., Wikipedia, OpenWebText)
    # dataset_CT = load_dataset(*args.dataset_CT, split="train")
    # with open(args.raw_CT, "wb") as f:
    #     pickle.dump(dataset_CT, f)
    with open(args.raw_CT, "rb") as f:
        dataset_CT = pickle.load(f)
    
    temp = [i.lower() for i in dataset_CT['text']]
    temp1 = [i.split(" ") for i in temp] # List[List[str]]
    wordStats = get_dataStats(temp, temp1, "ct", args) # 전체를 다 사용하지 못할 수도 있음. (filter wordStats)

    print(wordStats)


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
        num_cores_train = 40

    )
    
    # word = "example"  # Replace with the word you want to check
    pos_tagged = nltk.pos_tag(["a", "the"])
    print(pos_tagged)
    main(args)
