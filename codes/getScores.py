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

# def _select_samples(dataframe, ratio, notExist_sents):
def _select_samples(word, idx_score, ratio, data, max_len):

    # idx_score = 나중에 수정하자 score distribution이 이상하니까 아래 0.05부분은 버리자 
    lim = sum(idx_score['diff']) * ratio
    cur = 0
    res = -1
    for idx, row in idx_score.iterrows():
        if cur < lim:
            cur += row['diff']
            res += 1
        else:
            break
    selected = idx_score.iloc[:res]['index']
    len_selected = len(selected)
    
    exist_sents = data[selected]
    notExist_sents = data[data.index.difference(idx_score['index'])].sample(n=(max_len - len_selected), random_state=42)

    return exist_sents, notExist_sents


def get_scores(data, data_splitted, wordStats, pplModel, pplTokenizer, replaceToken, types, args): # List[str]
    pplTokenizer.add_special_tokens({"pad_token": replaceToken})

    data = pd.Series(data)
    data_splitted = pd.Series([set(i) for i in data_splitted])
    wordStats = wordStats # sampled dataframe will come

    print("extracting included sentence per token...")
    with ParallelUtils() as parallelUtils:
        parallelUtils.change_function(partial(_get_includeData, data = data_splitted))
        # series[(series[bools])]
        data_filtered = parallelUtils.do_series(wordStats['word'], pre_assign = False, num_cores = 10)
        data_filtered.name = "sents" # series[series[boolean(indeces)]]
         # 왜 이렇게 되지?
    idx_filtered = data_filtered.map(lambda x: x[x == True].index)
    idx_filtered.name = 'index'

    print("get original and masked sentences...")
    d = pd.concat([wordStats, idx_filtered], axis = 1)
    data_filtered_original = d.apply(lambda x: data[x['index']].to_list(), axis = 1) # series[str]
    data_filtered_masked = d.apply(lambda x: [re.sub(f" {x['word']} ", f" {replaceToken} ", sent) for sent in data[x['index']]], axis = 1) # series[str]
    data_filtered_original.name = "original"
    
    # 각각의 단어에 대해 
    def tokenize(examples, tokenizer = pplTokenizer):
        kwargs = {"padding": "max_length", "truncation": True, "max_length": 512, "return_tensors": "pt"}
        return tokenizer(examples['text'], **kwargs)
    
    for word, idxes, original_sents, masked_sents in tqdm(zip(wordStats['word'], idx_filtered, data_filtered_original, data_filtered_masked)):
        print(f"{word} is being processing...")
        ids = pd.Series(list(idxes))
        ids.name = "index"
        original_sents = [i for i in original_sents if not pd.isna(i)]
        masked_sents = [i for i in masked_sents if not pd.isna(i)]

        print(sum([i == None for i in tqdm(original_sents)]))
        original_sents = Dataset.from_dict({"text": original_sents}).map(tokenize, num_proc = args.num_cores, batched = True).with_format("torch")
        masked_sents = Dataset.from_dict({"text": masked_sents}).map(tokenize, num_proc = args.num_cores, batched = True).with_format("torch")
        original_sents = original_sents.remove_columns("text")
        masked_sents = masked_sents.remove_columns('text')

        scores_original = pplModel.get_perplexity(input_texts = original_sents, batch = 32)
        
        with open(args.scores_original_CT.format(word), "wb") as f:
            pickle.dump(scores_original, f)
        
        scores_masked = pplModel.get_perplexity(input_texts = masked_sents, batch = 32)
        
        with open(args.scores_masked_CT.format(word), "wb") as f:
            pickle.dump(scores_masked, f)



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
        
    token_list = ['and', 'one', 'the', 'for', 'new', 'time', 'they', 'was', 'has', 'that', 'who', 'when']
    token_list = ['and', 'one', 'the', 'for', 'new', 'time', 'they', 'was', 'has', 'that', 'who']
    
    wordStats_bools = wordStats.apply(lambda x: x['word'] in token_list, axis = 1)
    wordStats = wordStats[wordStats_bools]
    # max_sample_size = int(len(temp) * wordStats['rat'].min())
    # print(max_sample_size)

    pplParams = {"torch_dtype": torch.bfloat16}
    pplModel = lmppl.LM(args.checkpoint_pplModel, **pplParams)
    pplTokenizer = AutoTokenizer.from_pretrained(args.checkpoint_pplModel)

    replaceToken = "<<PAD>>" 
    
    # LM scores(BERT류에는 적합하지 않을 수도 있음)
    get_scores(temp, temp1, wordStats, pplModel, pplTokenizer, replaceToken, "ct", args)



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
