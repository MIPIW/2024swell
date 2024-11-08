from datetime import datetime
from collections import Counter
import sys, os, pandas as pd
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
def ppl_dataset():
    pass

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

    
    out = {i: (j, k) for i, j, k in zip(data_tok, data_pos, data_cnt)}

    if types == "ct":
        with open(args.file_wordStats_CT, "wb") as f:
            pickle.dump(out, f)
    if types == "ft":
        with open(args.file_wordStats_FT, "wb") as f:
            pickle.dump(out, f)
    
    return out

def get_dataAvgLength(data): # List[List[str]]
    print(data[0])
    x = [len(i) for i in data]
    return sum(x) / len(x)


webhook_url = "https://hooks.slack.com/services/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"
@slack_sender(webhook_url=webhook_url, channel="mine")
def main(args):

    # Load a large corpus dataset for pretraining (e.g., Wikipedia, OpenWebText)
    # dataset_CT = load_dataset(*args.dataset_CT, split="train")
    dataset_FT = load_dataset(args.dataset_FT)

    # baseTokenizer = AutoTokenizer.from_pretrained(args.checkpoint_baseModel)
    # baseModel = AutoModelForMaskedLM.from_pretrained(args.checkpoint_baseModel)
    avgLength = get_dataAvgLength(dataset_FT['test']['tokens'])
    wordStats = get_dataStats(dataset_FT['test']['tokens'], "ft", args)

    # pplModel = lmppl.LM(args.checkpoint_pplModel)
    # pplCheckpoint = AutoTokenizer.from_pretrained(args.checkpoint_pplModel)

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
        ratio_CT = [0, 0.2, 0.4, 0.6, 0.8, 1.0],
        ratio_FT = [0, 0.2, 0.4, 0.6, 0.8, 1.0],
        checkpoint_CTModel = "weights/CT_{}",
        checkpoint_FTModel = "weights/FC_{}_{}",
        num_cores = 20
    )
    
    # word = "example"  # Replace with the word you want to check
    # pos_tagged = nltk.pos_tag([word])[0][1]
    # print(pos_tagged)

    main(args)

