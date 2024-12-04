from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import os, glob
from functools import partial
from argparse import Namespace
import pickle
from torch.utils.data import DataLoader

def _preprocess_data(examples, tokenizer, task):
        # Tokenize inputs and align labels
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding="max_length", max_length=128)
        labels = []
        
        for i, label in enumerate(examples[task]):  # replace with "pos_tags" for POS tagging
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word in the original sentence
            label_ids = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)  # Use -100 to ignore in loss computation
                elif word_idx != previous_word_idx:  # Only label the first token of a word
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

def tokenize_dataset(args, dataset, tokenizer, task, preFunc = _preprocess_data):
# Preprocess dataset
    
    # Apply preprocessing
    preFunc = partial(preFunc, tokenizer = tokenizer, task = task)
    tokenized_datasets = dataset.map(preFunc, batched=True, num_proc = args.num_cores)

    return tokenized_datasets


def set_trainer_testSingleToken(args, tokenized_datasets, tokenizer, model, task, lookup_token):

    lookup_token_id = tokenizer.convert_tokens_to_ids(lookup_token)
    print(lookup_token, lookup_token_id)


    def collate_fn(batch):
        input_ids = torch.stack([torch.tensor(x['input_ids']) for x in batch])
        attention_mask = torch.stack([torch.tensor(x['attention_mask']) for x in batch])
        labels = torch.tensor([x['labels'] for x in batch])
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

    batch_size = 512
    train_loader = DataLoader(tokenized_datasets['test'], batch_size=batch_size, collate_fn=collate_fn)

    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)
            
            idxForLookupToken = (input_ids == lookup_token_id).view(-1)

            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
                        
            labels = labels.view(-1)
            predictions = predictions.view(-1)

            labels = torch.logical_and(labels, idxForLookupToken).to(args.device)
            predictions = torch.logical_and(predictions, idxForLookupToken).to(args.device)

            for i in range(len(labels)):
                if labels[i] == 0:
                    continue
                else:
                    total += 1

                    if labels[i] == predictions[i]:
                        correct += 1


            # total += len(labels)
            # correct += (predictions == labels).sum().item()
    if total == 0:
        return 0

    return correct / total


def main(args):
    
    if os.path.isfile(args.raw_FT):
        with open(args.raw_FT, "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = load_dataset("conll2003")  # replace with "universal_dependencies" for POS tagging
        with open(args.raw_FT, "wb") as f:
            pickle.dump(dataset, f)

    f_o = open(args.eval_output_file, "w", encoding="utf-8")
    print("Evaluation Results per Each Tokens", sep="\t", file=f_o) # print header

    # evaluation
    for task in args.tasks:

        label_list = dataset["train"].features[task].feature.names  # or "pos_tags" for POS tagging

        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_baseModel, add_prefix_space = True)
        dataset = tokenize_dataset(args, dataset, tokenizer, task)
        
        
        for val in ['count', 'ppl']:
            for word in args.words:
                for ratio in args.ratio:
                
                    # evaluation
                    checkpoint = args.checkpoint_FTModel.format(task, word, val, ratio)
                    checkpoint = glob.glob(checkpoint + os.sep + "checkpoint-*")[0]
                    model = AutoModelForTokenClassification.from_pretrained(checkpoint, num_labels=len(label_list))
                    model.to(args.device)
                    res = set_trainer_testSingleToken(args, dataset, tokenizer, model, task, word)

                    print(f"{checkpoint.split('/')[-2]}:\t{res}", file=f_o) # print header

    f_o.close()



if __name__ == "__main__":


    words = ['one', 'for', 'new', 'time', 'they', 'was', 'has', 'that', 'who']
    words = ['they', 'was', 'has', 'that', 'who'] # selected words

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
        num_cores_train = 10,

        tasks = ["pos_tags", "dep_tags"],
        device = "cuda" if torch.cuda.is_available() else "cpu",
        eval_output_file = "/home/hyohyeongjang/2024SWELL/evalOutput.txt"
    )

    args.words = words


    main(args)