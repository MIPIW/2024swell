import yaml, pickle, re
from argparse import Namespace
import pandas as pd, numpy as np
from collections import Counter
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import Dataset
from functools import partial
from tqdm import tqdm
from datasets import load_dataset, Dataset, concatenate_datasets
import torch 
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

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
    open_close_dic = config['analysisMeta']['analysis_pos']

    trainRes = pd.DataFrame(columns = config['dataStats']['correctXPos'])
    evalResId = pd.DataFrame(columns = config['dataStats']['correctXPos'])
    evalResOOD = pd.DataFrame(columns = config['dataStats']['correctXPos'])
    evalResTd = pd.DataFrame(columns = config['dataStats']['correctXPos'])
    
    fName = config['analysisMeta']['analysis3']
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

def _tokenize_and_align_labels(examples, tokenizer, args, config):
    
    tokenized_inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512, is_split_into_words = True)
    
    labels = []
    for i, label in enumerate(examples["label"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(config['dataStats']['labelToId'][label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        # if unPos_idx is not None:
        # if pos_idx is not None:
        #     label_ids = [j if j == pos_idx else -100 for j in label_ids]

        labels.append(label_ids)
    
    tokenized_inputs["label"] = labels
    
    return tokenized_inputs

def token_frequency_macro(args, config):
    open_close_dic = config['analysisMeta']['analysis_pos']
    tokenizer = AutoTokenizer.from_pretrained(config['contTrain']['checkpoint_baseModel'], add_prefix_space=True)
    func = partial(_tokenize_and_align_labels, tokenizer = tokenizer, args = args, config = config)
    
    with open(config['contFiles']['train_dataset_CT'], "rb") as f:
        dataset_dic = pickle.load(f)

    ss = pd.DataFrame()
    for key in dataset_dic:
        x = dataset_dic[key]
        x.columns = ['text', 'label']
        x = Dataset.from_pandas(x)
        x = x.map(func, batched = True, num_proc=5)

        lst_word = [(ii, yaml_config['dataStats']['idToLabel'][jj]) for i, j in zip(x['input_ids'], x['label']) for ii, jj in zip(i, j) if jj != -100]
        
        word_count = Counter(lst_word)
        val_nn_unique = 0
        val_in_unique = 0
        val_nn = 0
        val_in = 0

        val_other_open_unique = 0
        val_other_open = 0
        val_other_close_unique = 0
        val_other_close = 0

        for key, value in word_count.items():

            if key[1] == "NN":
                val_nn += value
                val_nn_unique += 1

            if key[1] == "IN":
                val_in += value
                val_in_unique += 1
            
            if open_close_dic[key[1]] == "OPEN" and key[1] != "NN":
                val_other_open += value
                val_other_open_unique += 1
            
            if open_close_dic[key[1]] == "CLOSE" and key[1] != "IN":
                val_other_close += value
                val_other_close_unique += 1
        
        s = pd.DataFrame([val_nn, val_nn_unique, val_in, val_in_unique, val_other_open, val_other_open_unique, val_other_close, val_other_close_unique])
        
        ss = pd.concat([ss, s], axis = 1)
    
    ss = ss.T
    ss.index = range(len(dataset_dic))
    ss.columns = ['nn', 'nnu', 'in', 'inu', 'oo', 'oou', 'oc', 'ocu']

    ss.to_csv(config['analysisMeta']['dup_dic_macro'], index = False)
    

def token_frequency_micro(args, config):
    open_close_dic = config['analysisMeta']['analysis_pos']
    tokenizer = AutoTokenizer.from_pretrained(config['contTrain']['checkpoint_baseModel'], add_prefix_space=True)
    func = partial(_tokenize_and_align_labels, tokenizer=tokenizer, args=args, config=config)

    with open(config['contFiles']['train_dataset_CT'], "rb") as f:
        dataset_dic = pickle.load(f)

    ss = pd.DataFrame()
    for key in dataset_dic:
        x = dataset_dic[key]
        x.columns = ['text', 'label']
        x = Dataset.from_pandas(x)
        x = x.map(func, batched = True, num_proc=5)

        # Initialize lists to store duplication counts for each sentence
        nn_duplication_counts = []
        in_duplication_counts = []
        other_open_duplication_counts = []
        other_close_duplication_counts = []

        for input_ids, labels in zip(x['input_ids'], x['label']):
            # Create a list of (token, label) pairs for each sentence, ignoring -100 labels
            lst_word = [(ii, yaml_config['dataStats']['idToLabel'][jj]) for ii, jj in zip(input_ids, labels) if jj != -100]
            
            # Count occurrences of (token, label) pairs within the sentence
            word_count = Counter(lst_word)
            # Count duplicates per POS tag
            nn_count = [count for (token, tag), count in word_count.items() if tag == "NN"]
            in_count = [count for (token, tag), count in word_count.items() if tag == "IN"]
            other_open_count = [count for (token, tag), count in word_count.items() if open_close_dic.get(tag) == "OPEN" and tag != "NN"]
            other_close_count = [count for (token, tag), count in word_count.items() if open_close_dic.get(tag) == "CLOSE" and tag != "IN"]

            nn_count = sum(nn_count) / len(nn_count) if len(nn_count) > 0 else 0
            in_count = sum(in_count) / len(in_count) if len(in_count) > 0 else 0
            other_open_count = sum(other_open_count) / len(other_open_count) if len(other_open_count) > 0 else 0
            other_close_count = sum(other_close_count) / len(other_close_count) if len(other_close_count) > 0 else 0

            # Append duplication counts for this sentence
            nn_duplication_counts.append(nn_count)
            in_duplication_counts.append(in_count)
            other_open_duplication_counts.append(other_open_count)
            other_close_duplication_counts.append(other_close_count)

        # Calculate average duplication counts across all sentences
        val_nn_avg = sum(nn_duplication_counts) / len(nn_duplication_counts) if nn_duplication_counts else 0
        val_in_avg = sum(in_duplication_counts) / len(in_duplication_counts) if in_duplication_counts else 0
        val_other_open_avg = sum(other_open_duplication_counts) / len(other_open_duplication_counts) if other_open_duplication_counts else 0
        val_other_close_avg = sum(other_close_duplication_counts) / len(other_close_duplication_counts) if other_close_duplication_counts else 0

        # Create a DataFrame for the averages
        s = pd.DataFrame([{
            'NN': val_nn_avg,
            'IN': val_in_avg,
            'OPEN': val_other_open_avg,
            'CLOSED': val_other_close_avg
        }])

        # Concatenate results
        ss = pd.concat([ss, s], axis=0, ignore_index=True)
    
    ss.to_csv(config['analysisMeta']['dup_dic_micro'], index = False)

    

def get_dup_ratio(args, config, isMacro):
    if isMacro:
        ss = pd.read_csv(config['analysisMeta']['dup_dic_macro'])
        ss = pd.concat([ss['nn'] / ss['nnu'], ss['in'] / ss['inu'], ss['oo'] / ss['oou'], ss['oc'] / ss['ocu']], axis = 1)
        ss = ss.mean(axis = 0)
        ss.index = ['NN', 'IN', 'OPEN', "CLOSED"]
    else:
        ss = pd.read_csv(config['analysisMeta']['dup_dic_micro'])
        ss = ss.mean(axis = 0)


    return ss    



def debug_collate_fn(batch):
    
    labels = []
    mask = []
    ids = []
    for i, item in enumerate(batch):
        labels.append(item['label'])
        mask.append(item['attention_mask'])
        ids.append(item['input_ids'])
    
    return {"label": torch.tensor(labels), "input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}
                    


def get_f1_scores(args, config):
    bucket = 2
    num_labels = 39  # Number of classes for token classification
    NUM_EVAL_BUCKET = 5
    NUM_SUBBUCKET = 50

    tokenizer = AutoTokenizer.from_pretrained(config['contTrain']['checkpoint_baseModel'], add_prefix_space=True)

    with open(config['contFiles']['train_dataset_CT'], "rb") as f:
        dataset_dic = pickle.load(f)

    eval_range = list(range(bucket * 10, (bucket) * 10 + NUM_EVAL_BUCKET))
    for key, value in tqdm(dataset_dic.items()):
        value.columns = ['text', 'label']
        dataset_dic[key] = Dataset.from_pandas(value)

    eval_dataset = concatenate_datasets([value for key, value in dataset_dic.items() if key in eval_range])
    print("tokenizing datasets...")
    f = partial(_tokenize_and_align_labels, tokenizer = tokenizer, args = args, config = config)

    eval_dataset = eval_dataset.map(f, batched=True, num_proc = 10).remove_columns("text")
    eval_dataloader = DataLoader(eval_dataset, batch_size=config['contTrain']['batch_size'], collate_fn=debug_collate_fn, num_workers = 4, pin_memory = True, )

    for epoch in range(15):
        model = AutoModelForTokenClassification.from_pretrained(config['contTrain']['checkpoint_CTModel'].format(epoch, bucket, "2e-05"))

        # Initialize lists for storing true labels and predictions
        true_labels = []
        pred_labels = []

        # Loop through the evaluation dataset and make predictions
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            inputs = {key: batch[key] for key in batch if key in ["input_ids", "attention_mask"]}
            labels = batch['label']

            with torch.no_grad():
                outputs = model(**inputs)

            # Get the predicted class indices
            predictions = np.argmax(outputs.logits, axis=-1)

            # Align predictions and true labels (considering subword tokens)
            for pred, label, mask in zip(predictions, labels, inputs['attention_mask']):
                true_seq = [l for l, m in zip(label, mask) if l != -100]
                pred_seq = [p for p, l, m in zip(pred, label, mask) if l != -100]

                true_labels.extend(true_seq)
                pred_labels.extend(pred_seq)

        # Calculate the F1 score
        f1 = f1_score(true_labels, pred_labels, average='weighted')
        print(f"Epoch {epoch}: F1 Score = {f1:.4f}")


def main(args, config):

    # get_result_dataframe(args, config)
    get_f1_scores(args,config)
    # token_frequency_micro(args, config)
    # ss = get_dup_ratio(args, config, False)
    # print(ss)


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
    yaml_config['analysisMeta']['analysis_pos'] = {"'"+"'" if key == "'" else key : value for key, value in yaml_config['analysisMeta']['analysis_pos'].items()}
    
    main(args, yaml_config)
    