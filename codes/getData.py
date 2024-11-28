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


def set_trainer(dataset, model, tokenizer):
    conti_dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
    # conti_dataset = [example for example in tqdm(conti_dataset['text']) if "for" not in example]
    conti_dataset = [example for example in tqdm(conti_dataset['text'])]
    
    # dic = dict()
    # for example in tqdm():
    #     if "for" in example:
    #         c = Counter([(i, "for") for i in example])
    #         for key in c:
    #             if dic.get(key, None) is None:
    #                 dic[key] = c[key]
    #             else:
    #                 dic[key] = dic[key] + c[key]

    # Tokenize dataset
    def tokenize_function(example):
        return tokenizer(example['text'], return_tensors="pt", truncation=True, padding="max_length", max_length=512)

    # Tokenize dataset using map
    conti_dataset = Dataset.from_dict({"text": conti_dataset})
    tokenized_datasets = conti_dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    tokenized_datasets.set_format(type = "torch")

    
    # tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=30, remove_columns=["text"])

    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # Define pretraining arguments
    training_args = TrainingArguments(
        output_dir="/home/hyohyeongjang/2024aut_comprac/weights/roberta-pretrained-notFiltered",
        overwrite_output_dir=True,
        num_train_epochs=1,  # Adjust for longer pretraining
        per_device_train_batch_size=32,  # Adjust for available hardware
        save_steps=10000,
        logging_steps=10000,
        save_total_limit=3,
        prediction_loss_only=True,
        learning_rate=5e-5,
        weight_decay=0.01,
    )

    # Initialize Trainer for MLM
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets,
    )

    # Start pretraining
    return trainer


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


def make_samples(data, data_splitted, wordStats, pplModel, pplTokenizer, replaceToken, types, args): # List[str]
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

        scores_original = pplModel.get_perplexity(input_texts = original_sents, batch = 64)
        
        with open(args.scores_original_CT.format(word), "wb") as f:
            pickle.dump(scores_original, f)
        
        scores_masked = pplModel.get_perplexity(input_texts = masked_sents, batch = 64)
        
        with open(args.scores_masked_CT.format(word), "wb") as f:
            pickle.dump(scores_masked, f)

        # with open(args.scores_original_CT.format(word), "rb") as f:
        #     scores_original = pickle.load(f)

        # with open(args.scores_masked_CT.format(word), "rb") as f:
        #     scores_masked = pickle.load(f)

        # scores_diff = pd.Series((np.array(scores_masked) - np.array(scores_original)))
        # scores_diff.name = "diff"
        # print(scores_diff)

        # idx_score = pd.concat([ids, scores_diff], axis = 1).sort_values(by = "diff")

        # # 각각의 선발비율에 대해
        # for ratio in tqdm(args.ratio):
            
        #     exist, notExist = _select_samples(word, idx_score, ratio, data, len(scores_diff))

        #     if types == "ft":
        #         with open(args.data_FT.format(word, ratio), "wb") as f:
        #             pickle.dump([exist, notExist], f)

        #     if types == "ct":
        #         with open(args.data_CT.format(word, ratio), "wb") as f:
        #             pickle.dump([exist, notExist], f)



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
    # x = wordStats.sort_values(by = 'rat').groupby("pos")
    # for k, v in x:
    #     print(v)
    #     print()


    #############################################################
    # # CC, CD, DT, IN, JJ, MD, NN, PRP$, RB, RP, VBD, WDT
    # token_list = ['and', 'one', 'the', 'for', 'red', 'can', 'art', 'her', 'not', 'out', 'was', 'which']
    token_list = ['and', 'one', 'the', 'for', 'new', 'time', 'they', 'was', 'has', 'that', 'who', 'when']
    token_list = ['for', 'time', 'has'] # when 은 mask만 하면 됨. 
    
    wordStats_bools = wordStats.apply(lambda x: x['word'] in token_list, axis = 1)
    wordStats = wordStats[wordStats_bools]
    max_sample_size = int(len(temp) * wordStats['rat'].min())
    print(max_sample_size)

    pplParams = {"torch_dtype": torch.bfloat16}
    pplModel = lmppl.LM(args.checkpoint_pplModel, **pplParams)
    pplTokenizer = AutoTokenizer.from_pretrained(args.checkpoint_pplModel)

    replaceToken = "<<PAD>>" 
    
    make_samples(temp, temp1, wordStats, pplModel, pplTokenizer, replaceToken, "ct", args)
    #############################################################


    # dataset_FT = load_dataset(args.dataset_FT)
    # with open(args.raw_FT, "wb") as f:
    #     pickle.dump(dataset_FT, f)
    # with open(args.raw_FT, "rb") as f:
    #     dataset_FT = pickle.load(f)
        
    # temp = [" ".join(i) for i in dataset_FT['test']['tokens']] # List[str]
    # temp1 = [i.split(" ") for i in temp] # List[List[str]]
    # avgLength = get_dataAvgLength(temp1)
    # wordStats = get_dataStats(temp, "ct", args) # 전체를 다 사용하지 못할 수도 있음. (filter wordStats)
    # wordStats = wordStats.groupby(by = 'pos').sample(n = 1, random_state = 42)

    # with open(args.file_wordStats_CT, "rb") as f:
    #     x = pickle.load(f)
    # print([i[1] for i in x.groupby("pos")])

    # pplModel = lmppl.LM(args.checkpoint_pplModel)
    # pplTokenizer = AutoTokenizer.from_pretrained(args.checkpoint_pplModel)
    
    # replaceToken = "<PAD>>" # 임시
    # make_samples(temp, wordStats, pplModel, replaceToken, "ct", args)
    

    # trainer = set_trainer(dataset, model, tokenizer)
    # trainer.train()




if __name__ == "__main__":

    # already downloaded
    # nltk.download('wordnet')
    # nltk.download('averaged_perceptron_tagger')

    args = Namespace(
        dataset_CT = ("wikitext", "wikitext-103-v1"),
        dataset_FT = "conll2003",
        raw_CT = "/home/hyohyeongjang/2024SWELL/data_raw/ct_raw.pk",
        raw_FT = "/home/hyohyeongjang/2024SWELL/data_raw/ft_raw.pk",
        file_wordStats_CT = "/home/hyohyeongjang/2024SWELL/meta/word_ct.pk",
        file_wordStats_FT = "/home/hyohyeongjang/2024SWELL/meta/word_ft.pk",
        scores_original_CT = "/home/hyohyeongjang/2024SWELL/scores/score_CT_original_{}.pk",
        scores_masked_CT = "/home/hyohyeongjang/2024SWELL/scores/score_CT_mask_{}.pk",
        checkpoint_baseModel = "FacebookAI/roberta-base",
        checkpoint_pplModel = "meta-llama/Meta-Llama-3-8B-Instruct",
        ratio = [0, 0.2, 0.4, 0.6, 0.8, 1.0],
        data_CT = "/home/hyohyeongjang/2024SWELL/data_CT/CT_{}_{}.pk",
        data_FT = "/home/hyohyeongjang/2024SWELL/data_FT/FT_{}_{}.pk",
        checkpoint_CTModel = "/home/hyohyeongjang/2024SWELL/weights/CT_{}_{}",
        checkpoint_FTModel = "/home/hyohyeongjang/2024SWELL/weights/FT_{}_{}_{}",
        # num_cores = 40, # preprocessing
        bag_of_word_CT = "/home/hyohyeongjang/2024SWELL/meta/BOW_CT.pk",
        bag_of_word_FT = "/home/hyohyeongjang/2024SWELL/meta/BOW_FT.pk",
        cnt_pos_CT = "/home/hyohyeongjang/2024SWELL/meta/cnt_pos_CT.pk",
        cnt_pos_FT = "/home/hyohyeongjang/2024SWELL/meta/cnt_pos_FT.pk",
        num_cores = 40 # making dataset

    )
    
    # word = "example"  # Replace with the word you want to check
    pos_tagged = nltk.pos_tag(["a", "the"])
    print(pos_tagged)
    main(args)
