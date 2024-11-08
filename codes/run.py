from datetime import datetime
from collections import Counter
import sys, os, pandas as pd, numpy as np
from functools import reduce, partial
import pickle
from multiprocessing import Pool
from argparse import Namespace

from knockknock import slack_sender
from tqdm import tqdm
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
# device = "cuda" if torch.cuda.is_available() else "cpu"
# class MyLM(lmppl.LM):
#     """ Language Model. """

#     def __init__(self,
#                  model: str = 'gpt2',
#                  use_auth_token: bool = False,
#                  max_length: int = None,
#                  num_gpus: int = None,
#                  torch_dtype=None,
#                  device_map: str = None,
#                  low_cpu_mem_usage: bool = False,
#                  trust_remote_code: bool = True,
#                  offload_folder: str = None,
#                  hf_cache_dir: str = None):
#         """ Language Model.

#         @param model: Model alias or path to local model file.
#         @param use_auth_token: Huggingface transformers argument of `use_auth_token`
#         @param device: Device name to load the models.
#         @param num_gpus: Number of gpus to be used.
#         """
#         logging.info(f'Loading Model: `{model}`')

#         # load model
#         # params = {"local_files_only": not internet_connection(), "use_auth_token": use_auth_token,
#         params = {"use_auth_token": use_auth_token, "trust_remote_code": trust_remote_code}
#         if hf_cache_dir is not None:
#             params["cache_dir"] = hf_cache_dir
#         if offload_folder is not None:
#             params["offload_folder"] = offload_folder
#         self.tokenizer = transformers.AutoTokenizer.from_pretrained(model, **params)
#         self.config = transformers.AutoConfig.from_pretrained(model, **params)

#         params.update({"config": self.config, "low_cpu_mem_usage": low_cpu_mem_usage})
#         if torch_dtype is not None:
#             params['torch_dtype'] = torch_dtype
#         if device_map is not None:
#             params['device_map'] = device_map
#         self.model = transformers.AutoModelForCausalLM.from_pretrained(model, **params)

#         self.pad_token_initialized = False
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.add_special_tokens({'pad_token': "<<PAD>>"})
#             self.model.resize_token_embeddings(len(self.tokenizer))
#             self.pad_token_initialized = True

#         if max_length is None:
#             self.max_length = None
#         else:
#             self.max_length = max_length if max_length is not None else self.tokenizer.model_max_length
#             assert self.max_length <= self.tokenizer.model_max_length, f"{self.max_length} > {self.tokenizer.model_max_length}"

#         # loss function
#         self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

#         # GPU setup
#         self.device = self.model.device
#         print(self.device)
#         if device_map is None:
#             num_gpus = torch.cuda.device_count() if num_gpus is None else num_gpus
#             if num_gpus == 1:
#                 self.model.to('cuda')
#                 self.device = self.model.device
#             elif num_gpus > 1:
#                 self.model = torch.nn.DataParallel(self.model)
#                 self.model.to('cuda')
#                 self.device = self.model.module.device
#         self.model.eval()
#         logging.info(f'\t * model is loaded on: {self.device}')



# def set_trainer(dataset, model, tokenizer):
#     conti_dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
#     # conti_dataset = [example for example in tqdm(conti_dataset['text']) if "for" not in example]
#     conti_dataset = [example for example in tqdm(conti_dataset['text'])]
    
#     # dic = dict()
#     # for example in tqdm():
#     #     if "for" in example:
#     #         c = Counter([(i, "for") for i in example])
#     #         for key in c:
#     #             if dic.get(key, None) is None:
#     #                 dic[key] = c[key]
#     #             else:
#     #                 dic[key] = dic[key] + c[key]

#     # Tokenize dataset
#     def tokenize_function(example):
#         return tokenizer(example['text'], return_tensors="pt", truncation=True, padding="max_length", max_length=512)

#     # Tokenize dataset using map
#     conti_dataset = Dataset.from_dict({"text": conti_dataset})
#     tokenized_datasets = conti_dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
#     tokenized_datasets.set_format(type = "torch")

    
#     # tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=30, remove_columns=["text"])

#     # Data collator for MLM
#     data_collator = DataCollatorForLanguageModeling(
#         tokenizer=tokenizer, mlm=True, mlm_probability=0.15
#     )

#     # Define pretraining arguments
#     training_args = TrainingArguments(
#         output_dir="/home/hyohyeongjang/2024aut_comprac/weights/roberta-pretrained-notFiltered",
#         overwrite_output_dir=True,
#         num_train_epochs=1,  # Adjust for longer pretraining
#         per_device_train_batch_size=32,  # Adjust for available hardware
#         save_steps=10000,
#         logging_steps=10000,
#         save_total_limit=3,
#         prediction_loss_only=True,
#         learning_rate=5e-5,
#         weight_decay=0.01,
#     )

#     # Initialize Trainer for MLM
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         data_collator=data_collator,
#         train_dataset=tokenized_datasets,
#     )

#     # Start pretraining
#     return trainer
def get_dataAvgLength(data): # List[List[str]]
    x = [len(i) for i in data]
    return sum(x) / len(x)

def _filter_data(datum): # str
    if datum == None:
        return {}
    x = set(datum.split(" "))
    x = set(filter(lambda x: 2 < len(x) < 10, x))

    return x    

def _is_included(token, data, total_length):
    return sum([token in datum for datum in data]) / total_length

def get_dataStats(data, types, args): # List[str]

    if types == "ct" and os.path.isfile(args.file_wordStats_CT):
        with open(args.file_wordStats_CT, "wb") as f:
            out = pickle.load(f)
            return out    
    if types == "ft" and os.path.isfile(args.file_wordStats_FT):
        with open(args.file_wordStats_FT, "rb") as f:
            out = pickle.load(f)
            return out

    with ParallelUtils() as parallelUtils:
        parallelUtils.change_function(_filter_data)
        # get unigram counts    
        data_filtered = pd.Series(data)
        data_filtered = parallelUtils.do_series(data_filtered, pre_assign = False, num_cores = args.num_cores)
        data_filtered = data_filtered.to_list() # series[set[str(tokens)]]
        data_filtered = list(reduce(lambda x, y : x | y, data_filtered)) # list[str(tokens)]
    
    print("-------------------", data_filtered)
    chunk_size = len(data_filtered) // args.num_cores
    with Pool(processes=args.num_cores) as pool:
        total_length = len(data)
        chunks = [data_filtered[i * chunk_size:(i + 1) * chunk_size] for i in range(args.num_cores - 1)]
        chunks.append(data_filtered[(args.num_cores - 1) * chunk_size:])
        data_cnt = pool.map(partial(_is_included, data = data, total_length = total_length), chunks)
        data_pos = pool.map(nltk.pos_tag, chunks)
    
    data_cnt = list(filter(lambda x: 0.4 < x[1] < 0.6, enumerate(data_cnt)))
    
    data_pos = [data_pos[i[0]] for i in data_cnt]
    data_tok = [data_filtered[i[0]] for i in data_cnt]
    data_cnt = [data_cnt[i[1]] for i in data_cnt]

    out = pd.DataFrame({i: (j, k) for i, j, k in zip(data_tok, data_pos, data_cnt)}).T.reset_index()
    out.columns = ["word", 'pos', 'rat']

    if types == "ct":
        with open(args.file_wordStats_CT, "wb") as f:
            pickle.dump(out, f)
    if types == "ft":
        with open(args.file_wordStats_FT, "wb") as f:
            pickle.dump(out, f)
    
    return out


def _get_includeData(tokenInfo, data): # token: str, data: List[str]

    exist_sents = data.map(lambda x: tokenInfo['word'] in x)

    return exist_sents

# def _select_samples(dataframe, ratio, notExist_sents):
def _select_samples(word, idx_score, ratio, data):

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
    notExist_sents = data[~idx_score['index']].sample(n=len_selected, random_state=42)

    return exist_sents, notExist_sents


def get_existSamples(data, wordStats, pplModel, replaceToken, args): # List[str]

    data = pd.Series(data)
    wordStats = wordStats # sampled dataframe will come
    with ParallelUtils() as parallelUtils:
        parallelUtils.change_function(partial(_get_includeData, data = data))
        # series[(series[bools])]
        data_filtered = parallelUtils.do_dataFrame(wordStats, axis = 1, pre_assign = False, num_cores = args.num_cores)
        data_filtered.name = "sents" # boolean(indeces)

    idx_filtered = data_filtered.map(lambda x: x[x == True].index)
    d = pd.concat([wordStats, data_filtered], axis = 1)
    data_filtered_original = d.apply(lambda x: data[x['sents']], axis = 1) # series[str]
    data_filtered_masked = d.apply(lambda x: [sent.replace(x['word'], replaceToken) for sent in data[x['sents']]], axis = 1).explode() # series[str]
    data_filtered_original.name = "original"

    # 각각의 단어에 대해 
    for word, idxes, original_sents, masked_sents in zip(wordStats['word'], idx_filtered, data_filtered_original, data_filtered_masked):

        scores_original = pplModel.get_perplexity(input_texts = original_sents)
        scores_masked = pplModel.get_perplexity(input_texts = masked_sents)
        scores_diff = pd.Series((np.array(scores_masked) - np.array(scores_original)))
        scores_diff.name = "diff"

        idx_score = pd.concat([idxes, scores_diff], axis = 1).sort_values(by = "diff")
        
        # 각각의 선발비율에 대해
        for ratio in args.ratio:

    x = pd.concat([x, scores_diff, data_filtered_original], axis = 1).sort_values(by=['word', "diff"])
    x = x.groupby("word")

    for ratio in args.ratio:
        out = x.agg(partial(_select_samples, ratio = ratio)) # ser[(str, List[str], int)]

    return out
    # 하 이제 어떻게 하지

def get_trainingData(singleTokenInfo, notExistData):
    token, sents, lens = singleTokenInfo

    
    

    


    




webhook_url = "https://hooks.slack.com/services/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"
@slack_sender(webhook_url=webhook_url, channel="mine")
def main(args):

    # Load a large corpus dataset for pretraining (e.g., Wikipedia, OpenWebText)
    # dataset_CT = load_dataset(*args.dataset_CT, split="train")
    # dataset_FT = load_dataset(args.dataset_FT)
    # with open("x.pk", "wb") as f:
    #     pickle.dump(dataset_FT, f)
    with open("x.pk", "rb") as f:
        dataset_FT = pickle.load(f)
    
    
    # baseTokenizer = AutoTokenizer.from_pretrained(args.checkpoint_baseModel)
    # baseModel = AutoModelForMaskedLM.from_pretrained(args.checkpoint_baseModel)
    temp = [" ".join(i) for i in dataset_FT['test']['tokens']] # List[str]
    temp1 = [i.split(" ") for i in temp] # List[List[str]]
    avgLength = get_dataAvgLength(temp1)
    wordStats = get_dataStats(temp, "ft", args) # 전체를 다 사용하지 못할 수도 있음. (filter wordStats)
    print(wordStats.head())

    # pplModel = lmppl.LM(args.checkpoint_pplModel)
    # pplCheckpoint = AutoTokenizer.from_pretrained(args.checkpoint_pplModel)
    pplModel = None
    replaceToken = "<pad>"
    sampleSents = get_existSamples(temp, wordStats, pplModel, replaceToken, args)

    for e, tokenInfo in sampleSents.iterrows():
        trainingData_perToken = get_trainingData(tokenInfo, temp)
    
    

    # trainer = set_trainer(dataset, model, tokenizer)
    # trainer.train()



if __name__ == "__main__":

    # already downloaded
    # nltk.download('wordnet')
    # nltk.download('averaged_perceptron_tagger')

    args = Namespace(
        dataset_CT = ("wikitext", "wikitext-103-v1"),
        dataset_FT = "conll2003",
        file_wordStats_CT = "/home/hyohyeongjang/2024SWELL/meta/word_ct.pk",
        file_wordStats_FT = "/home/hyohyeongjang/2024SWELL/meta/word_ft.pk",
        checkpoint_baseModel = "FacebookAI/roberta-base",
        checkpoint_pplModel = "meta-llama/Meta-Llama-3-8B-Instruct",
        ratio = [0, 0.2, 0.4, 0.6, 0.8, 1.0],
        checkpoint_CTModel = "weights/CT_{}",
        checkpoint_FTModel = "weights/FC_{}_{}",
        num_cores = 20
    )
    
    # word = "example"  # Replace with the word you want to check
    # pos_tagged = nltk.pos_tag([word])[0][1]
    # print(pos_tagged)

    main(args)

