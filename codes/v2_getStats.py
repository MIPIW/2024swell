import yaml, pickle
from argparse import ArgumentParser, Namespace
# import nltk
from tqdm import tqdm
tqdm.pandas()
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_scheduler
from datasets import load_dataset, Dataset
import os, sys
from multiprocessing import Pool
from collections import Counter, defaultdict
from functools import partial
from torch.utils.data import DataLoader
import pandas as pd, numpy as np
import string
import torch, random
import logging
from torch.utils.data import DataLoader, DistributedSampler

# sys.path.append(os.path.expanduser('~/'))
# from myUtils.parallelUtils import ParallelUtils
from knockknock import slack_sender

def setup_distributed():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

def cleanup_distributed():
    dist.destroy_process_group()


def set_random_seed(seed: int):
    # Set seed for Python's built-in random module
    random.seed(seed)
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for PyTorch (CPU)
    torch.manual_seed(seed)
    
    # Set seed for PyTorch (GPU, if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Ensure deterministic behavior in PyTorch operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the seed for reproducibility
SEED = 42
set_random_seed(SEED)


# Helper function to process a chunk of data
def _process_chunk(chunk):
    return [item.translate(str.maketrans('', '', string.punctuation)).lower() for item in chunk]

def get_raw(args, config, types, n_shard_s, n_shard_e):
    if types == "ct":
        # if not os.path.isfile(config['originalData']['raw_CT']):
        #     dataset_CT = load_dataset(*config['originalData']['dataset_CT'])

        #     with open(config['originalData']['raw_CT'], "wb") as f:
        #         pickle.dump(dataset_CT, f)

        #     original = [i.lower() for i in dataset_CT['text']]
        #     splitted = [i.split(" ") for i in original] # List[List[str]]
            
        #     orisplit = [(i,j) for i, j in zip(original, splitted) if 256 < len(j) < 512]
        #     original = [i[0] for i in orisplit]
        #     splitted = [i[1] for i in orisplit]

        #     with open(config['originalData']['raw_filter_original_CT'], "wb") as f:
        #         pickle.dump(original, f)

        #     with open(config['originalData']['raw_filter_split_CT'], "wb") as f:
        #         pickle.dump(splitted, f)
        if False:
            pass
        else:
            # with open(config['originalData']['raw_CT'], "rb") as f:
            #     dataset_CT = pickle.load(f)
            data = {}
            for i in range(n_shard_s, n_shard_e+1):
                print(f"opening checkpoint {i}")
                with open(config['originalData']['raw_CT']+f"_{i}.pk", "rb") as f:
                    data[i] = pickle.load(f)
                    
                    # Calculate chunk sizes
                    chunk_size = (len(data[i]) + config['dataStats']['num_cores'] - 1) // config['dataStats']['num_cores']  # Ensures all data points are covered
                    chunks = [data[i][j:j + chunk_size] for j in range(0, len(data[i]), chunk_size)]
                    
                    with Pool(config['dataStats']['num_cores']) as pool:
                        # Use pool.imap for parallel processing with progress tracking
                        processed_chunks = list(tqdm(pool.imap(_process_chunk, chunks), total=len(chunks)))
                    
                    # Flatten the list of lists back into a single list
                    data[i] = [item for sublist in processed_chunks for item in sublist]

            return data
            
            # with open(config['originalData']['raw_filter_original_CT'], "rb") as f:
            #     original = pickle.load(f)

            # with open(config['originalData']['raw_filter_split_CT'], "rb") as f:
            #     splitted = pickle.load(f)


            # with open(config['originalData']['raw_filter_original_CT'], "rb") as f:
            #     original = pickle.load(f)

            # with open(config['originalData']['raw_filter_split_CT'], "rb") as f:
            #     splitted = pickle.load(f)

    # if types == "ft":

    #     if not os.path.isfile(config['originalData']['raw_FT']):
    #         dataset_FT = load_dataset(*config['originalData']['dataset_FT'])
    #         with open(config['originalData']['raw_FT'], "wb") as f:
    #             pickle.dump(dataset_FT, f)

    #     else:
    #         with open(config['originalData']['raw_FT'], "rb") as f:
    #             dataset_FT = pickle.load(f)

    #     return dataset_FT, None


# Function to process a chunk of data
def _process_chunk2(chunk):
    return [item.split(" ") for item in chunk]

def split_data(args, config, original_data, n_shard_s, n_shard_e):

    splitted_data = {}
    for i in range(n_shard_s, n_shard_e+1):
        # Calculate chunk sizes
        print(f"splitting checkpoint {i}")

        chunk_size = (len(original_data[i]) + config['dataStats']['num_cores'] - 1) // config['dataStats']['num_cores']  # Ensures all data points are covered
        chunks = [original_data[i][j:j + chunk_size] for j in range(0, len(original_data[i]), chunk_size)]
        
        with Pool(config['dataStats']['num_cores']) as pool:
            # Use pool.imap for parallel processing with progress tracking
            processed_chunks = list(tqdm(pool.imap(_process_chunk2, chunks), total=len(chunks)))
        
        # Flatten the list of lists back into a single list
        splitted_data[i] = [item for sublist in processed_chunks for item in sublist]

    return splitted_data

def _process_chunk3(chunk):
    return nltk.pos_tag_sents(chunk)

def _get_pos(args, config, splitted_data):

    # Calculate chunk sizes
    chunk_size = (len(splitted_data) + config['dataStats']['num_cores'] - 1) // config['dataStats']['num_cores']  # Ensures all data points are covered
    chunks = [splitted_data[j:j + chunk_size] for j in range(0, len(splitted_data), chunk_size)]
    
    with Pool(config['dataStats']['num_cores']) as pool:
        # Use pool.imap for parallel processing with progress tracking
        processed_chunks = list(tqdm(pool.imap(_process_chunk3, chunks), total=len(chunks)))
    
    # Flatten the list of lists back into a single list
    return [item for sublist in processed_chunks for item in sublist]



def get_pos(args, config, splitted_data):

    if not os.path.isfile(config['dataStats']['cnt_pos_CT']):
        word_pos = [nltk.pos_tag(space_sents) for space_sents in tqdm(splitted_data)]    
        with open(config['dataStats']['cnt_pos_CT'], "wb") as f:
            pickle.dump(word_pos, f)
    else:
        with open(config['dataStats']['cnt_pos_CT'], "rb") as f:
            word_pos = pickle.load(f)
    
    # from list[tuple] to dict
    pos_dict = {i[0]: i[1] for i in word_pos[1]}
    # convert penn to upos
    pos_dict = {key: config['dataStats']['pennToUpos'].get(value, "X") for key, value in pos_dict.items()}

    return word_pos[0], pos_dict

def get_tokenizer(args, config):
    tokenizer = AutoTokenizer.from_pretrained(config['contTrain']['checkpoint_baseModel'])
    return tokenizer

def _tokenize(dataset, tokenizer):
    return {"text": [tokenizer.tokenize(sent, truncation = True, max_length = 512) for sent in dataset['text']]}

def get_subword_pos(args, config, original, splitted, pos_dict, tokenizer):

    if os.path.isfile(config['dataStats']['subwordPos_CT']):
        with open(config['dataStats']['subword_CT'], "rb") as f:
            subword_valid = pickle.load(f)

        with open(config['dataStats']['subwordPos_CT'], "rb") as f:
            subword_pos = pickle.load(f)
        
        return subword_valid, subword_pos
        
    else:       
        
        original = Dataset.from_dict({"text": original})
        subwords = original.map(partial(_tokenize, tokenizer = tokenizer), batched = True, num_proc=config['dataStats']['num_cores'])
        subwords = subwords['text']

        # filter too short ones


        # Generate POS tags for subwords
        subword_pos = []
        subword_valid = []
        for sentence_split, sentence_subwords in tqdm(zip(splitted, subwords)):
            
            if len(sentence_subwords) < 256:
                continue

            pos_list = []
            split_idx = 0  # Index to track the current word in `splitted`
            
            for subword in sentence_subwords:
                # Check if the subword starts a new word (space-indicated subwords or initial word)
                if subword.startswith("Ġ") or (split_idx < len(sentence_split) and pos_list == []):
                    current_word = sentence_split[split_idx]
                    pos_tag = pos_dict.get(current_word, "UNK")  # Default POS tag if not found
                    split_idx += 1
                # Assign the current POS tag to the subword
                pos_list.append(pos_tag)
            
            subword_pos.append(pos_list)
            subword_valid.append(sentence_subwords)
        
        with open(config['dataStats']['subwordPos_CT'], "wb") as f:
            pickle.dump(subword_pos, f)
        
        with open(config['dataStats']['subword_CT'], "wb") as f:
            pickle.dump(subword_valid, f)

        return subword_valid, subword_pos

def get_pos_stats(args, config, subword_pos, subwords):

    if os.path.isfile(config['dataStats']['pos_freq_CT']):
        with open(config['dataStats']['pos_freq_CT'], "rb") as f:
            pos_frequency_list = pickle.load(f)

        with open(config['dataStats']['pos_abund_CT'], "rb") as f:
            pos_abundancy_list = pickle.load(f)
        
        return pos_frequency_list, pos_abundancy_list

    pos_frequency_list = []
    pos_abundancy_list = []

    for sentWords, sentPos in zip(subwords, subword_pos):
        # Map each POS to the set of unique subwords associated with it
        pos_to_words = defaultdict(set)

        for word, pos in zip(sentWords, sentPos):
            pos_to_words[pos].add(word)

        # Count the number of unique tokens for each POS
        pos_frequency = {pos: len(words) for pos, words in pos_to_words.items()}
        pos_abundancy = {pos: len(set(words)) for pos, words in pos_to_words.items()}
        
        pos_frequency_list.append(pos_frequency)
        pos_abundancy_list.append(pos_abundancy)
    
    with open(config['dataStats']['pos_freq_CT'], "wb") as f:
        pickle.dump(pos_frequency_list, f)
        
    with open(config['dataStats']['pos_abund_CT'], "wb") as f:
        pickle.dump(pos_abundancy_list, f)

    return pos_frequency_list, pos_abundancy_list

def get_dataset(args, config, subwords, pos_freq, pos_abund):
    
    pos_list = ['NOUN','PART', 'NUM', 'SCONJ','ADJ','ADP','DET','CCONJ','PROPN','PRON','ADV','INTJ','VERB','AUX']
    df = pd.DataFrame([subwords, pos_freq, pos_abund]).T
    df.columns = ['subwords', "freq", "abund"]

    for pos in tqdm(pos_list):
        
        pos_frequency = df.apply(lambda x: x['freq'].get(pos, 0), axis = 1)
        pos_abundancy = df.apply(lambda x: x['abund'].get(pos, 0), axis = 1)
        pos_abundancy =  pos_abundancy / (pos_frequency + 1e-6)

        context_frequency = df.apply(lambda x: sum([val for key, val in x['freq'].items() if key != pos and key in pos_list]), axis = 1)
        context_abundancy = df.apply(lambda x: sum([val for key, val in x['abund'].items() if key != pos and key in pos_list]), axis = 1)
        context_abundancy = context_abundancy / (context_frequency + 1e-6)

        pos_freq_avg = pos_frequency.mean()
        pos_abund_avg = pos_abundancy.mean()
        context_freq_avg = context_frequency.mean()
        context_abund_avg = context_abundancy.mean()


        pos_frequency = pos_frequency.map(lambda x: x > pos_freq_avg)
        pos_abundancy = pos_abundancy.map(lambda x: x > pos_abund_avg)
        context_frequency = context_frequency.map(lambda x: x > context_freq_avg)
        context_abundancy = context_abundancy.map(lambda x: x > context_abund_avg)

        df_new = pd.concat([df, pos_frequency, pos_abundancy, context_frequency, context_abundancy], axis = 1)
        df_new.columns = ['subwords', 'freq', 'abund', 'pf', 'pa', 'cf', 'ca']
        
        print(pos)
        print(df_new.groupby(['pf', 'pa']).count(), "\n", df_new.groupby(['cf', 'ca']).count())


def debug_collate_fn(batch):
    
    labels = []
    mask = []
    ids = []
    for i, item in enumerate(batch):
        labels.append(item['label'])
        mask.append(item['attention_mask'])
        ids.append(item['input_ids'])
    
    return {"label": torch.tensor(labels), "input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}
                    


def _get_stats1(x):
    return Counter([tag for _, tag in x])

def _get_stats2(x):
    return Counter([tag for _, tag in set(x)])

def _process_chunk4(pos, df):
    is_nonZero = df[pos].map(lambda x: x != 0)
    quant_pos = df.loc[is_nonZero, pos].quantile(np.arange(0.1, 1.1, 0.1))

    return {pos: [quant_pos[i] for i in np.arange(0.1, 1.1, 0.1)]}

def tokenize_and_align_labels(examples, args, config, tokenizer, unPos_idx, pos_idx):
    
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

def compute_label_accuracies(preds, labels, num_labels):
    # Flatten predictions and labels, ignore pad tokens (-100)
    preds_flat = np.argmax(preds, axis=-1).flatten()
    labels_flat = labels.flatten()
    
    # Mask to ignore padding tokens (-100)
    mask = labels_flat != -100
    
    # Filter out padding tokens
    preds_flat = preds_flat[mask]
    labels_flat = labels_flat[mask]
    
    # Track correct predictions and counts per label
    label_correct = defaultdict(int)
    label_total = defaultdict(int)
    
    for pred, label in zip(preds_flat, labels_flat):
        label_total[label] += 1
        if pred == label:
            label_correct[label] += 1
    
    # Calculate accuracy per label
    label_accuracies = {}
    for label in range(num_labels):
        if label_total[label] > 0:
            label_accuracies[label] = label_correct[label] / label_total[label]
        else:
            label_accuracies[label] = None  # No instances of this label in the batch
    
    return label_accuracies

def reduce_dict(args, counts_dict, num_labels):
    for label in range(num_labels):
        tensor = torch.tensor(counts_dict[label], dtype=torch.float32).to(args.local_rank)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        counts_dict[label] = tensor.item()


webhook_url = "https://hooks.slack.com/services/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"
@slack_sender(webhook_url=webhook_url, channel="mine")
def main(args, config):
    

    ## preprocessing
    ############### # split dataset into shard
    # dataset_CT = load_dataset(*config['originalData']['dataset_CT'])

    # with open("asdf", "wb") as f:
    #     pickle.dump(dataset_CT, f)

    # with open("asdf", "rb") as f:
    #     dataset_CT = pickle.load(f)

    # n = len(dataset_CT['train']['text'])
    # nn = n // 10
    # for i in range(0, 10):
    #     with open(config['originalData']['raw_CT']+f"_{i}.pk", "wb") as f:
    #         if i < 9:
    #             x = dataset_CT['train']['text'][i * nn: (i+1) * nn]
    #         else:
    #             x = dataset_CT['train']['text'][i * nn:]

    #         pickle.dump(x, f)

    # ############## # get pos tagged per each shard, per each sentence
    # for i in range(10):
    #     data_CT_lower = get_raw(args, config, "ct", i, i)
    #     data_CT_splitted = split_data(args, config, data_CT_lower, i, i)

    #     d1 = data_CT_lower[i]
    #     d2 = data_CT_splitted[i]
    #     ds = [(k, j) for k, j in zip(d1, d2) if 256 < len(j) < 512] 

    #     d1 = [k[0] for k in ds]
    #     d2 = [k[1] for k in ds]
        # d3 = _get_pos(args, config, d2)

        # with open(config['originalData']['raw_CT']+f"_{i}_process_d1.pk", "wb") as f:
        #     pickle.dump(d1, f)

        # with open(config['originalData']['raw_CT']+f"_{i}_process_d2.pk", "wb") as f:
        #     pickle.dump(d2, f)

        # with open(config['originalData']['raw_CT']+f"_{i}_process_d3.pk", "wb") as f:
        #     pickle.dump(d3, f)
       

    # # get pos stats
    # ############### get pos count dict(temporary, to be converted into dataframe)
    # for i in range(10):
    #     print(f"reading processed dataset {i}")
    #     with open(config['originalData']['raw_CT']+f"_{i}_process_d3.pk", "rb") as f:
    #         d3 = pickle.load(f)

    #     d3 = pd.Series(d3) 

    #     # with ParallelUtils() as parallelUtils:
    #     d3_normal_count = d3.progress_map(_get_stats1)
    #     d3_unique_count = d3.progress_map(_get_stats2)

    #         # parallelUtils.change_function(_get_stats1)    
    #         # d3_normal_count = parallelUtils.do_series(d3, pre_assign = False, num_cores = config['dataStats']['num_cores'])        # get frequency, abundancy information`
    #         # parallelUtils.change_function(_get_stats2)    
    #         # d3_unique_count = parallelUtils.do_series(d3, pre_assign = False, num_cores = config['dataStats']['num_cores'])        # get frequency, abundancy information`


    #     print(f"saving POS dataset {i}")
    #     with open(config['dataStats']['pos_freq_CT']+f"_{i}.pk", "wb") as f:
    #         pickle.dump([d3_normal_count, d3_unique_count], f)
    

    ############### get pos total count from dict
    # x = pd.DataFrame()    
    # y = pd.DataFrame()
    # for i in tqdm(range(10)):
    #     print(f"reading processed dataset {i}")

    #     with open(config['dataStats']['pos_freq_CT']+f"_{i}.pk", "rb") as f:
    #         df_normal_count, df_unique_count = pickle.load(f)
    #         x = pd.concat([x, df_normal_count], axis = 0)
    #         y = pd.concat([y, df_unique_count], axis = 0)
        
    # freq = {}
    # for i, v in tqdm(x.iterrows()):
    #     for k, val in v[0].items():
    #         if freq.get(k, 0) == 0:
    #             freq[k] = val
    #         else:
    #             freq[k] += val                

    # ddupFreq = {}
    # for i, v in tqdm(y.iterrows()):
    #     for k, val in v[0].items():
    #         if ddupFreq.get(k, 0) == 0:
    #             ddupFreq[k] = val
    #         else:
    #             ddupFreq[k] += val                

    
    # x1 = pd.DataFrame.from_dict(freq, orient='index')
    # x2 = pd.DataFrame.from_dict(ddupFreq, orient='index')
    # x1.to_csv(config['dataStats']['posStats'], index = True)
    # x2.to_csv(config['dataStats']['ddupPosStats'], index = True)

    # considering number of pos in total, we can classify LL, LH, HL, HH groups of POS
    # have to decide which of them is adequate to extract




    # # get pos frequency per each sentences, and get quatile 
    # ################ convert pos count dict to dataframe
    # # # # 특정한 token이 있다고 치고
    # FULL_POS = config['dataStats']['correctXPos']
    # pos_normal = pd.DataFrame(columns=FULL_POS)
    # pos_unique = pd.DataFrame(columns=FULL_POS)

    # for i in tqdm(range(10)):
    #     with open(config['dataStats']['pos_freq_CT']+f"_{i}.pk", "rb") as f:
    #         df_normal_count, df_unique_count = pickle.load(f)

    #     df_normal_count = [dict(i) for i in df_normal_count]
    #     df_normal_count = [{key: i.get(key, 0) for key in FULL_POS} for i in df_normal_count]
    #     df_unique_count = [dict(i) for i in df_unique_count]
    #     df_unique_count = [{key: i.get(key, 0) for key in FULL_POS} for i in df_unique_count]

    #     df_normal_count = pd.DataFrame(df_normal_count)
    #     df_unique_count = pd.DataFrame(df_unique_count)
    #     df_normal_count.to_csv(config['dataStats']['pos_freq_CT']+f"_{i}_normal.csv", index = False)
    #     df_unique_count.to_csv(config['dataStats']['pos_freq_CT']+f"_{i}_unique.csv", index = False)        


    ############## # get qualtile data statisics per POS
    # FULL_POS = config['dataStats']['correctXPosNoSym']
    # # comparative adjective, wh determiner, gerund or present participle, prersonal pronoun
    # # respectively LL, LH, HL, HH
    # pos_normal = pd.DataFrame(columns=FULL_POS)
    # pos_unique = pd.DataFrame(columns=FULL_POS)

    # for i in tqdm(range(10)):
    #     df_normal_count = pd.read_csv(config['dataStats']['pos_freq_CT']+f"_{i}_normal.csv")
    #     df_unique_count = pd.read_csv(config['dataStats']['pos_freq_CT']+f"_{i}_unique.csv")

    #     pos_normal = pd.concat([pos_normal, df_normal_count], axis = 0)
    #     pos_unique = pd.concat([pos_unique, df_unique_count], axis = 0)
    
    # pos_normal = pos_normal.reset_index(drop = True)
    # pos_unique = pos_unique.reset_index(drop = True)

    # dic = {}

    # with Pool(34) as pool:
    #     # Use pool.imap for parallel processing with progress tracking
    #     f = partial(_process_chunk4, df = pos_normal)
    #     processed_chunks = list(tqdm(pool.imap(f, FULL_POS), total=len(FULL_POS)))
    
    # # Flatten the list of lists back into a single list
    # dic = {key: value for pos_item in processed_chunks for key, value in pos_item.items() }
    # pd.DataFrame.from_dict(dic, orient='index').to_csv(config['dataStats']['posQuantiles'], index = True)
    
    ##########################################다른스레드에서 여기까지 실행###########################################
    # considering number of sentences in each bins, with LL, LH, HL, HH settings
    # comparative adjective, wh determiner, gerund or present participle, prersonal pronoun
    # respectively LL, LH, HL, HH
    # FINAL_TARGET_POS = ['JJR', 'WDT','MD', "VBP"] # decided by these steps
    FINAL_TARGET_POS = ['JJR', 'WDT'] # decided by these steps
    
    FINAL_RANGE = [(0,1), (2,4), (5,7)]

               

    # ## get file index to be extracted(filter file index to control)  
    ##########################################################################################
    ##################여기에 다른 토큰들 개수를 통제하는 로직 더해져야 함#########################
    ##########################################################################################
    
    
    # ############## filter those which have even tokens
    # FULL_POS = config['dataStats']['correctXPosNoSym']
    
    # pos_normal = pd.DataFrame(columns=FULL_POS)
    # pos_unique = pd.DataFrame(columns=FULL_POS)

    # for i in tqdm(range(10)):
    #     df_normal_count = pd.read_csv(config['dataStats']['pos_freq_CT']+f"_{i}_normal.csv")
    #     df_unique_count = pd.read_csv(config['dataStats']['pos_freq_CT']+f"_{i}_unique.csv")

    #     pos_normal = pd.concat([pos_normal, df_normal_count], axis = 0).reset_index(drop = True)
    #     pos_unique = pd.concat([pos_unique, df_unique_count], axis = 0).reset_index(drop = True)

    # n_lowQ = {}
    # n_highQ = {}
    # u_lowQ = {}
    # u_highQ = {}
    # for pos in FINAL_TARGET_POS:
    #     # target_low_off = pos_normal[pos].map(lambda x: ranges[0][0] <= x <= ranges[0][1] )
    #     target_low_idx = pos_normal[pos].map(lambda x: FINAL_RANGE[1][0] <= x <= FINAL_RANGE[1][1] )
    #     target_high_idx = pos_normal[pos].map(lambda x: FINAL_RANGE[2][0] <= x <= FINAL_RANGE[2][1] )
    #     # target_high_off = pos_normal[pos].map(lambda x: ranges[3][0] <= x <= ranges[3][1] )
        
    #     n_tgt_low = pos_normal[target_low_idx]
    #     n_tgt_high = pos_normal[target_high_idx]
    #     u_tgt_low = pos_unique[target_low_idx]
    #     u_tgt_high = pos_unique[target_high_idx]

    #     n_tgt_low_noPos = n_tgt_low.drop(columns = pos, inplace = False)
    #     n_tgt_high_noPos = n_tgt_high.drop(columns = pos, inplace = False)
    #     u_tgt_low_noPos = u_tgt_low.drop(columns = pos, inplace = False)
    #     u_tgt_high_noPos = u_tgt_high.drop(columns = pos, inplace = False)

    #     n_stdLow = n_tgt_low_noPos.std(axis = 1)
    #     n_stdHigh = n_tgt_high_noPos.std(axis = 1)
    #     u_stdLow = u_tgt_low_noPos.std(axis = 1)
    #     u_stdHigh = u_tgt_high_noPos.std(axis = 1)

    #     n_lowQ[pos] = n_stdLow.quantile(np.arange(0.1, 1.1, 0.1))
    #     n_highQ[pos] = n_stdHigh.quantile(np.arange(0.1, 1.1, 0.1))
    #     u_lowQ[pos] = u_stdLow.quantile(np.arange(0.1, 1.1, 0.1))
    #     u_highQ[pos] = u_stdHigh.quantile(np.arange(0.1, 1.1, 0.1))

    # pd.DataFrame.from_dict(n_lowQ, orient = "index").to_csv(config['dataStats']['otherPosQuantilesLowNormal'])
    # pd.DataFrame.from_dict(n_highQ, orient = "index").to_csv(config['dataStats']['otherPosQuantilesHighNormal'])
    # pd.DataFrame.from_dict(u_lowQ, orient = "index").to_csv(config['dataStats']['otherPosQuantilesLowUnique'])
    # pd.DataFrame.from_dict(u_highQ, orient = "index").to_csv(config['dataStats']['otherPosQuantilesHighUnique'])
        

#    ############ filter those which have even tokens
#     FULL_POS = config['dataStats']['correctXPosNoSym']
    
#     pos_normal = pd.DataFrame(columns=FULL_POS)
#     pos_unique = pd.DataFrame(columns=FULL_POS)

#     for i in tqdm(range(10)):
#         df_normal_count = pd.read_csv(config['dataStats']['pos_freq_CT']+f"_{i}_normal.csv")
#         df_unique_count = pd.read_csv(config['dataStats']['pos_freq_CT']+f"_{i}_unique.csv")

#         pos_normal = pd.concat([pos_normal, df_normal_count], axis = 0).reset_index(drop = True)
#         pos_unique = pd.concat([pos_unique, df_unique_count], axis = 0).reset_index(drop = True)

#     for pos in FINAL_TARGET_POS:
#         if pos == FINAL_TARGET_POS[0]:
#             unPos = FINAL_TARGET_POS[1]
#         else:
#             unPos = FINAL_TARGET_POS[0]
        
#         # target_low_off = pos_normal[pos].map(lambda x: ranges[0][0] <= x <= ranges[0][1] )
#         target_low_idx = pos_normal[pos].map(lambda x: FINAL_RANGE[1][0] <= x <= FINAL_RANGE[1][1] )
#         target_high_idx = pos_normal[pos].map(lambda x: FINAL_RANGE[2][0] <= x <= FINAL_RANGE[2][1] )
#         # target_high_off = pos_normal[pos].map(lambda x: ranges[3][0] <= x <= ranges[3][1] )


#         n_tgt_low = pos_normal[target_low_idx]
#         n_tgt_high = pos_normal[target_high_idx]
#         u_tgt_low = pos_unique[target_low_idx]
#         u_tgt_high = pos_unique[target_high_idx]

#         # # filter by quantile
#         # n_tgt_low_noPos = n_tgt_low.drop(columns = pos, inplace = False)
#         # n_tgt_high_noPos = n_tgt_high.drop(columns = pos, inplace = False)
#         # u_tgt_low_noPos = u_tgt_low.drop(columns = pos, inplace = False)
#         # u_tgt_high_noPos = u_tgt_high.drop(columns = pos, inplace = False)

#         # n_stdLow = n_tgt_low_noPos.std(axis = 1)
#         # n_stdHigh = n_tgt_high_noPos.std(axis = 1)
#         # u_stdLow = u_tgt_low_noPos.std(axis = 1)
#         # u_stdHigh = u_tgt_high_noPos.std(axis = 1)

#         # n_stdLow_base = n_stdLow.quantile(0.1)
#         # n_stdLow_top = n_stdLow.quantile(0.9)
#         # n_stdHigh_base = n_stdHigh.quantile(0.1)
#         # n_stdHigh_top = n_stdHigh.quantile(0.9)
#         # u_stdLow_base = u_stdLow.quantile(0.1)
#         # u_stdLow_top = u_stdLow.quantile(0.9)
#         # u_stdHigh_base = u_stdHigh.quantile(0.1)
#         # u_stdHigh_top = u_stdHigh.quantile(0.9)

#         # n_stdLow_tgt = n_stdLow.map(lambda x: n_stdLow_base < x < n_stdLow_top).reindex(range(len(target_low_idx)), fill_value=False)
#         # n_stdHigh_tgt = n_stdHigh.map(lambda x: n_stdHigh_base < x < n_stdHigh_top).reindex(range(len(target_high_idx)), fill_value=False)
#         # u_stdLow_tgt = u_stdLow.map(lambda x: u_stdLow_base < x < u_stdLow_top).reindex(range(len(target_low_idx)), fill_value=False)
#         # u_stdHigh_tgt = u_stdHigh.map(lambda x: u_stdHigh_base < x < u_stdHigh_top).reindex(range(len(target_high_idx)), fill_value=False)

#         # filter by unPos
#         unMin_low = n_tgt_low[unPos] == 0
#         unMin_high = n_tgt_high[unPos] == 0
#         unMin_low_eval = n_tgt_low[unPos] >= 1
#         unMin_high_eval = n_tgt_high[unPos] >= 1

        
#         unMin_low = unMin_low.reindex(range(len(target_low_idx)), fill_value=False)
#         unMin_high = unMin_high.reindex(range(len(target_low_idx)), fill_value=False)
#         unMin_low_eval = unMin_low_eval.reindex(range(len(target_low_idx)), fill_value=False)
#         unMin_high_eval = unMin_high_eval.reindex(range(len(target_low_idx)), fill_value=False)
        
#         # eval_low_idx = pos_normal[target_low_idx & n_stdLow_tgt & u_stdLow_tgt & unMin_low_eval]
#         # eval_high_idx = pos_normal[target_high_idx & n_stdHigh_tgt & u_stdHigh_tgt & unMin_high_eval]
#         # target_low_idx = pos_normal[target_low_idx & n_stdLow_tgt & u_stdLow_tgt & unMin_low]
#         # target_high_idx = pos_normal[target_high_idx & n_stdHigh_tgt & u_stdHigh_tgt & unMin_high]

#         eval_low_idx = pos_normal[target_low_idx &  unMin_low_eval]
#         eval_high_idx = pos_normal[target_high_idx &  unMin_high_eval]
#         target_low_idx = pos_normal[target_low_idx &  unMin_low]
#         target_high_idx = pos_normal[target_high_idx &  unMin_high]

#         # # eval_low_idx = n_stdLow.map(lambda x: n_stdLow_top < x).reindex(range(len(target_low_idx)))
#         # # eval_high_idx = n_stdHigh.map(lambda x: n_stdHigh_top < x).reindex(range(len(target_high_idx)))

#         # eval_low_idx = pos_normal[target_low_idx & eval_low_idx]
#         # eval_high_idx = pos_normal[target_high_idx & eval_high_idx]
#         # target_low_idx = pos_normal[target_low_idx & n_stdLow_tgt & u_stdLow_tgt]
#         # target_high_idx = pos_normal[target_high_idx & n_stdHigh_tgt & u_stdHigh_tgt]
        

#         print(pos, len(target_low_idx), len(target_high_idx), len(eval_low_idx), len(eval_high_idx))

#         with open(config['contFiles']['data_CT'].format(pos, "low"), "wb") as f:
#             pickle.dump([target_low_idx, eval_low_idx], f)
        
#         with open(config['contFiles']['data_CT'].format(pos, "high"), "wb") as f:
#             pickle.dump([target_high_idx, eval_high_idx], f)



    # ############### extract data by idx
    # tot_former = 0
    
    # for i in tqdm(range(10)):

    #     with open(config['originalData']['raw_CT']+f"_{i}_process_d2.pk", "rb") as f:
    #         sent_lower = pickle.load(f)
    #         sent_lower = pd.Series(sent_lower)

    #     with open(config['originalData']['raw_CT']+f"_{i}_process_d3.pk", "rb") as f:
    #         sent_pos = pickle.load(f)
    #         sent_pos = pd.Series([i[1] for i in sent] for sent in sent_pos)
            
    #     sent_lower.index = pd.RangeIndex(start=tot_former, stop=tot_former+len(sent_lower))
    #     sent_pos.index = pd.RangeIndex(start=tot_former, stop=tot_former+len(sent_pos))

    #     print(sent_lower.head(), sent_pos.head(), len(sent_lower), len(sent_pos))
    
    #     for pos in tqdm(FINAL_TARGET_POS):
            
    #         with open(config['contFiles']['data_CT'].format(pos, "low"), "rb") as f:
    #             idx_low, eval_idx_low = pickle.load(f)
            
    #         with open(config['contFiles']['data_CT'].format(pos, "high"), "rb") as f:
    #             idx_high, eval_idx_high = pickle.load(f) 

    #         print(idx_low.head(), idx_high.head())
             
    #         idx_low = idx_low.loc[(idx_low.index > tot_former) & (idx_low.index < (tot_former + len(sent_lower)))].index
    #         idx_high = idx_high.iloc[(idx_high.index > tot_former) & (idx_high.index < (tot_former + len(sent_lower)))].index
    #         eval_idx_low = eval_idx_low.loc[(eval_idx_low.index > tot_former) & (eval_idx_low.index < (tot_former + len(sent_lower)))].index
    #         eval_idx_high = eval_idx_high.loc[(eval_idx_high.index > tot_former) & (eval_idx_high.index < (tot_former + len(sent_lower)))].index

    #         sent_low = sent_lower[idx_low]
    #         sent_high = sent_lower[idx_high]
    #         label_low = sent_pos[idx_low]
    #         label_high = sent_pos[idx_high]

    #         eval_sent_low = sent_lower[eval_idx_low]
    #         eval_sent_high = sent_lower[eval_idx_high]
    #         eval_label_low = sent_pos[eval_idx_low]
    #         eval_label_high = sent_pos[eval_idx_high]


    #         with open(config['contFiles']['data_CT_str'].format(i, pos, "low"), "wb") as f:
    #             pickle.dump([sent_low, label_low, eval_sent_low, eval_label_low], f)
            
    #         with open(config['contFiles']['data_CT_str'].format(i, pos, "high"), "wb") as f:
    #             pickle.dump([sent_high, label_high, eval_sent_high, eval_label_high], f)   

    #     tot_former += len(sent_lower)



    train_dataset_dic = {}
    eval_dataset_dic = {}
    eval_dataset_df = pd.DataFrame(columns = ['text', 'label'])

    for lowHigh in ['low', 'high']:
        for pos in tqdm(FINAL_TARGET_POS):    
            
            serData = pd.Series()
            serLabel = pd.Series()
            serEvaldata = pd.Series()
            serEvalLabel = pd.Series()

            for i in range(10):
                with open(config['contFiles']['data_CT_str'].format(i, pos, lowHigh), "rb") as f:
                    data, label, eval_data, eval_label = pickle.load(f)
                    
                serData = pd.concat([serData, data], ignore_index = True)
                serLabel = pd.concat([serLabel, label], ignore_index = True)
                serEvaldata = pd.concat([serEvaldata, eval_data], ignore_index = True)
                serEvalLabel = pd.concat([serEvalLabel, eval_label], ignore_index = True)
        
            serData = serData.reset_index(drop = True)
            serLabel = serLabel.reset_index(drop = True)
            serEvaldata = serEvaldata.reset_index(drop = True)
            serEvalLabel = serEvalLabel.reset_index(drop = True)

            data = pd.concat([serData, serLabel], axis = 1)
            eval_data = pd.concat([serEvaldata, serEvalLabel], axis = 1)
            data.columns = ['text', 'label']
            eval_data.columns = ['text', 'label']

            train_data = data.sample(n = min(50000, len(data)), random_state=42)
            eval_data = eval_data.sample(n = min(50000, len(eval_data)), random_state=42)
            
            train_dataset_dic[f"{pos}_{lowHigh}"] = train_data.reset_index(drop = True)
            eval_dataset_dic[f"{pos}_{lowHigh}"] = eval_data.reset_index(drop = True)
    


        
    IDed = pd.DataFrame(columns = ["text", 'label'])
    OODed = pd.DataFrame(columns= ['text', 'label'])

    for pos in tqdm(FINAL_TARGET_POS):   
        d1 = train_dataset_dic[f"{pos}_low"]
        d2 = train_dataset_dic[f"{pos}_high"]

        sampledD1 = d1.sample(n = 16384, random_state = 42).reset_index(drop = True)        
        posSerD1 = sampledD1.apply(lambda x: [i for i, j in zip(x['text'], x['label']) if j == pos], axis = 1)
        posSetD1 = set(posSerD1.explode().drop_duplicates().to_list())
        
        posSerD2 = d2.apply(lambda x: [i for i, j in zip(x['text'], x['label']) if j == pos], axis = 1)
        d2BothIdx = posSerD2.map(lambda x : all([i in posSetD1 for i in x]))
        sampledD2 = d2[d2BothIdx].sample(n = 16384, random_state = 42).reset_index(drop = True)

        t2 = sampledD2.apply(lambda x: [i for i, j in zip(x['text'], x['label']) if j == pos], axis = 1)
        t2 = set(t2.explode().drop_duplicates().to_list())
        
        print(len(posSetD1), len(t2), len(posSetD1 - t2), len(t2 - posSetD1))

        ed1 = eval_dataset_dic[f"{pos}_low"]
        ed2 = eval_dataset_dic[f"{pos}_high"]

        posSerEd1 = ed1.apply(lambda x: [i for i, j in zip(x['text'], x['label']) if j == pos], axis = 1)
        posSerEd2 = ed2.apply(lambda x: [i for i, j in zip(x['text'], x['label']) if j == pos], axis = 1)
        ed1BothIdx = posSerEd1.map(lambda x : all([i in posSetD1 for i in x]))
        ed2BothIdx = posSerEd2.map(lambda x : all([i in posSetD1 for i in x]))

        IDEd1 = ed1[ed1BothIdx].sample(n = min(16384, ed1BothIdx.sum())).reset_index(drop = True)
        OODEd1 = ed1[~ed1BothIdx].sample(n = min(16384, (~ed1BothIdx).sum())).reset_index(drop = True)
        IDEd2 = ed2[ed2BothIdx].sample(n = min(16384, ed2BothIdx.sum())).reset_index(drop = True)
        OODEd2 = ed2[~ed2BothIdx].sample(n = min(16384, (~ed2BothIdx).sum())).reset_index(drop = True)

        IDedTemp = pd.concat([IDEd1, IDEd2], axis = 0)
        OODedTemp = pd.concat([OODEd1, OODEd2], axis = 0)

        print(len(ed1), len(IDEd1), len(OODEd1))
        print(len(ed2), len(IDEd2), len(OODEd2))


        IDed = pd.concat([IDed, IDedTemp], axis = 0)
        OODed = pd.concat([OODed, OODedTemp], axis = 0)

    eval_dataset_id = Dataset.from_pandas(IDed.reset_index(drop = True))
    eval_dataset_od = Dataset.from_pandas(OODed.reset_index(drop = True))

    for key in train_dataset_dic:
        train_dataset_dic[key] = Dataset.from_pandas(train_dataset_dic[key])
     
    with open(config['contFiles']['train_dataset_CT'], "wb") as f:
        pickle.dump(train_dataset_dic, f)

    with open(config['contFiles']['eval_dataset_CT'], "wb") as f:
        pickle.dump([eval_dataset_id, eval_dataset_od], f)
    

    # ## train model
    # model_name = "FacebookAI/roberta-base"
    # num_labels = 39  # Number of classes for token classification
    # tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    # model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    # model.save_pretrained(config['contTrain']['checkpoint_CTModel'].format(0, "base"))

    # with open(config['contFiles']['train_dataset_CT'], "rb") as f:
    #     train_dataset_dic = pickle.load(f)

    # with open(config['contFiles']['eval_dataset_CT'], "rb") as f:
    #     eval_dataset_id, eval_dataset_od = pickle.load(f)

    # train_dataset = {}
    # for key, value in train_dataset_dic.items():
    #     pos = key.split("_")[0]
    #     unPos_idx = [item for item in FINAL_TARGET_POS if item != pos]
    #     unPos_idx = [config['dataStats']['labelToId'][i] for i in unPos_idx]

    #     f = partial(tokenize_and_align_labels, tokenizer = tokenizer, args = args, config = config, unPos_idx = None, pos_idx = config['dataStats']['labelToId'][pos])
    #     train_dataset[key] = value.map(f, batched=True, num_proc = config['contTrain']['num_cores_train'])

    # train_dataset = {key: val.remove_columns("text") for key, val in train_dataset.items()}

    # f = partial(tokenize_and_align_labels, tokenizer = tokenizer, args = args, config = config, unPos_idx = None, pos_idx = None)
    # eval_dataset = eval_dataset.map(f, batched=True, num_proc = config['contTrain']['num_cores_train'])
    # eval_dataset = eval_dataset.remove_columns('text')
        
    #     # Create DataLoaders
    # if ddp:
    #     train_dataloader = {key: DataLoader(val, sampler=DistributedSampler(val), batch_size=config['contTrain']['batch_size'], shuffle=False, collate_fn=debug_collate_fn) for key, val in train_dataset.items()}
    #     eval_dataloader = DataLoader(eval_dataset, sampler=DistributedSampler(eval_dataset), batch_size=config['contTrain']['batch_size'], collate_fn=debug_collate_fn)
    # else:
    #     train_dataloader = {key: DataLoader(val, batch_size=config['contTrain']['batch_size'], shuffle=True, collate_fn=debug_collate_fn) for key, val in train_dataset.items()}
    #     eval_dataloader = DataLoader(eval_dataset, batch_size=config['contTrain']['batch_size'], collate_fn=debug_collate_fn)

    # # Move model to GPU if available
    # model.to(args.local_rank)
    
    # if args.ddp:
    #     model = DDP(model, device_ids=[args.local_rank])

    # optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    # num_training_steps = sum([len(i) for i in train_dataloader]) * 10  # Assuming 3 epochs
    # lr_scheduler = get_scheduler(
    #     name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    # )

    # if args.ddp and dist.get_rank() == 0:
    #     logging.basicConfig(
    #         filename="trainingSteps.log", 
    #         format='%(asctime)s %(levelname)s:%(message)s',
    #         datefmt='%Y%m%d %H%M%S',
    #         level=logging.INFO
    #     )
    # elif not args.ddp:
    #     logging.basicConfig(
    #         filename="trainingSteps.log", 
    #         format='%(asctime)s %(levelname)s:%(message)s',
    #         datefmt='%Y%m%d %H%M%S',
    #         level=logging.INFO
    # )

    # num_epochs = 10


    # # baseline evaluation
    # eval_correct_count = defaultdict(int, {label: 0 for label in range(num_labels)})
    # eval_total_count = defaultdict(int, {label: 0 for label in range(num_labels)})

    # model.eval()

    # with torch.no_grad():
    #     progress_bar = tqdm(eval_dataloader, desc=f"evaluating...")

    #     for batch in progress_bar:
            
    #         batch = {"input_ids": batch['input_ids'].to(args.local_rank), "attention_mask": batch['attention_mask'].to(args.local_rank), "labels": batch['label'].to(args.local_rank)}
            
    #         outputs = model(**batch)
    #         logits = outputs.logits
            
    #         preds = logits.detach().cpu().numpy()
    #         labels = batch["labels"].detach().cpu().numpy()
    #         label_accuracies = compute_label_accuracies(preds, labels, num_labels)
        
    #         for label, acc in label_accuracies.items():
    #             if acc is not None:
    #                 eval_correct_count[label] += acc * len(np.where(labels == label)[0])
    #                 eval_total_count[label] += len(np.where(labels == label)[0])
        
    #     dist.barrier()

    #     if args.ddp:
    #         reduce_dict(args, eval_correct_count, num_labels)
    #         reduce_dict(args, eval_total_count, num_labels)

    #     dist.barrier()
    #     if (args.ddp and dist.get_rank() == 0) or not args.ddp:
    #         print("i will write this!")
    #         out = ""
    #         for label in range(num_labels):
    #             if eval_total_count[label] > 0:
    #                 acc = eval_correct_count[label] / eval_total_count[label]
    #                 out += f"{round(acc, 4)}    "
    #             else:
    #                 out += "X   "
                
    #             print(label, eval_correct_count[label], eval_total_count[label])

    #         logging.info("---eval accuracy at 0 of base--------------------------------------")        
    #         logging.info(out)
        
    #     dist.barrier()
    

    # for epoch in range(num_epochs):

    #     for e, subEpoch in enumerate([f"{i}_low" for i in FINAL_TARGET_POS] + [f"{i}_high" for i in FINAL_TARGET_POS]):
    #         print(subEpoch)

    #         model.train()
    #         if args.ddp:
    #             train_dataloader[subEpoch].sampler.set_epoch(epoch)
    #         progress_bar = tqdm(train_dataloader[subEpoch], desc=f"E: {epoch}, SE: {e}")
    #         total_loss = 0
    #         total_acc = 0
    #         step = 0
    #         label_correct_counts = defaultdict(int, {label: 0 for label in range(num_labels)})
    #         label_total_counts = defaultdict(int, {label: 0 for label in range(num_labels)})

    #         for batch in progress_bar:
                
    #             batch = {"input_ids": batch['input_ids'].to(args.local_rank), "attention_mask": batch['attention_mask'].to(args.local_rank), "labels": batch['label'].to(args.local_rank)}

    #             # Forward pass
    #             outputs = model(**batch)
    #             loss = outputs.loss
    #             logits = outputs.logits
    #             # Backward pass and optimization
    #             optimizer.zero_grad()
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 

    #             optimizer.step()
    #             lr_scheduler.step()

    #             # Compute per-label accuracy for this batch
    #             preds = logits.detach().cpu().numpy()
    #             labels = batch["labels"].detach().cpu().numpy()
    #             label_accuracies = compute_label_accuracies(preds, labels, num_labels)

    #             # Track total counts for each label
    #             for label, acc in label_accuracies.items():
    #                 if acc is not None:
    #                     label_correct_counts[label] += acc * len(np.where(labels == label)[0])
    #                     label_total_counts[label] += len(np.where(labels == label)[0])

    #             # Track total loss
    #             total_loss += loss.item()
    #             step += 1

    #             # Display loss and accuracy in the progress bar
    #             progress_bar.set_postfix(loss=loss.item())

    #         # Calculate average loss and accuracy per label
    #         avg_loss = total_loss / step
            
    #         dist.barrier()
            
    #         if args.ddp:
    #             reduce_dict(args, label_correct_counts, num_labels)
    #             reduce_dict(args, label_total_counts, num_labels)
                
    #         dist.barrier()

    #         if (args.ddp and dist.get_rank() == 0) or not args.ddp:
    #             out = ""
    #             for label in range(num_labels):
    #                 if label_total_counts[label] > 0:
    #                     acc = label_correct_counts[label] / label_total_counts[label]
    #                     out += f"{round(acc, 4)}    "
    #                 else:
    #                     out += "X   "
    #                 print(label, label_correct_counts[label], label_total_counts[label])

    #             logging.info(f"---training accuracy at {epoch} of {subEpoch}--------------------------------------")       
    #             logging.info(out)

    #         dist.barrier()


    #         # Validation loop
    #         eval_correct_count = defaultdict(int, {label: 0 for label in range(num_labels)})
    #         eval_total_count = defaultdict(int, {label: 0 for label in range(num_labels)})

    #         if not args.ddp or dist.get_rank() == 0:
    #             model.module.save_pretrained(config['contTrain']['checkpoint_CTModel'].format(epoch, subEpoch))

    #         model.eval()

    #         with torch.no_grad():
    #             progress_bar = tqdm(eval_dataloader, desc=f"evaluating...")
    #             for batch in progress_bar:
                    
    #                 batch = {"input_ids": batch['input_ids'].to(args.local_rank), "attention_mask": batch['attention_mask'].to(args.local_rank), "labels": batch['label'].to(args.local_rank)}
                    
    #                 outputs = model(**batch)
    #                 logits = outputs.logits
                    
    #                 preds = logits.detach().cpu().numpy()
    #                 labels = batch["labels"].detach().cpu().numpy()
    #                 label_accuracies = compute_label_accuracies(preds, labels, num_labels)
                
    #                 for label, acc in label_accuracies.items():
    #                     if acc is not None:
    #                         eval_correct_count[label] += acc * len(np.where(labels == label)[0])
    #                         eval_total_count[label] += len(np.where(labels == label)[0])

    #             dist.barrier()
                            
    #             if args.ddp:
    #                 reduce_dict(args, eval_correct_count, num_labels)
    #                 reduce_dict(args, eval_total_count, num_labels)
                            
    #             dist.barrier()
    #             if (args.ddp and dist.get_rank() == 0) or not args.ddp:
    #                 out = ""
    #                 for label in range(num_labels):
    #                     if eval_total_count[label] > 0:
    #                         acc = eval_correct_count[label] / eval_total_count[label]
    #                         out += f"{round(acc, 4)}    "
    #                     else:
    #                         out += "X   "
                    
    #                     print(label, eval_correct_count[label], eval_total_count[label])

    #                 logging.info(f"---eval accuracy at {epoch} of {subEpoch}--------------------------------------")        
    #                 logging.info(out)

    #             dist.barrier()


    

    # ############### filter those which have even duplicated tokens
    # FULL_POS = config['dataStats']['correctXPosNoSym']
    # # comparative adjective, wh determiner, gerund or present participle, prersonal pronoun
    # # respectively LL, LH, HL, HH
    # pos_normal = pd.DataFrame(columns=FULL_POS)
    # pos_unique = pd.DataFrame(columns=FULL_POS)

    # for i in tqdm(range(2)):
    #     df_normal_count = pd.read_csv(config['dataStats']['pos_freq_CT']+f"_{i}_normal.csv")
    #     df_unique_count = pd.read_csv(config['dataStats']['pos_freq_CT']+f"_{i}_unique.csv")

    #     pos_normal = pd.concat([pos_normal, df_normal_count], axis = 0).reset_index(drop = True)
    #     pos_unique = pd.concat([pos_unique, df_unique_count], axis = 0).reset_index(drop = True)


    # lowQ = {}
    # highQ = {}
    # for pos in FINAL_TARGET_POS:
    #     # target_low_off = pos_normal[pos].map(lambda x: ranges[0][0] <= x <= ranges[0][1] )
    #     target_low_idx = pos_normal[pos].map(lambda x: FINAL_RANGE[1][0] <= x <= FINAL_RANGE[1][1] )
    #     target_high_idx = pos_normal[pos].map(lambda x: FINAL_RANGE[2][0] <= x <= FINAL_RANGE[2][1] )
    #     # target_high_off = pos_normal[pos].map(lambda x: ranges[3][0] <= x <= ranges[3][1] )
        
    #     target_low = pos_normal[target_low_idx]
    #     target_high = pos_normal[target_high_idx]
    #     target_low_noPos = target_low.drop(columns = pos, inplace = False)
    #     target_high_noPos = target_high.drop(columns = pos, inplace = False)

    #     stdLow = target_low_noPos.std(axis = 0)
    #     stdHigh = target_high_noPos.std(axis = 0)

    #     lowQ[pos] = stdLow.quantile(np.arange(0.1, 1.1, 0.1)), 
    #     highQ[pos] = stdHigh.quantile(np.arange(0.1, 1.1, 0.1))

    # pd.DataFrame.from_dict(lowQ, orient = "index").to_csv(config['dataStats']['otherPosQuantilesLow'])
    # pd.DataFrame.from_dict(highQ, orient = "index").to_csv(config['dataStats']['otherPosQuantilesHigh'])

        




            


    # print("------load pretraining dataset------")
    # data_CT_lower, data_CT_splitted = get_raw(args, config, "ct")
    # print("------load finetuning dataset------")
    # data_FT, _ = get_raw(args, config, "ft")
    # # _ is word included ratio of CT dataset
    # print("------load tagged pos information------")
    # _, pos_dict = get_pos(args, config, data_CT_splitted) # data_splitted is optionally used
    # # print(Counter([value for key, value in pos_dict.items()]))
    # print("------get subword pos information------")
    # tokenizer = get_tokenizer(args, config)
    # print("------get subword pos information------")
    # subwords, subword_pos = get_subword_pos(args, config, data_CT_lower, data_CT_splitted, pos_dict, tokenizer)
    # print("------get subword pos freq/abund------")
    # pos_freq, pos_abund = get_pos_stats(args, config, subword_pos, subwords)
    # print(len(subwords), len(subword_pos), len(pos_freq), len(pos_abund))
    # get_dataset(args, config, subwords, pos_freq, pos_abund)


    # tokenizer = AutoTokenizer.from_pretrained(config['contTrain'])



    pass


if __name__ == "__main__":

    dp = False
    ddp = False

    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.utils.data import DistributedSampler
        import torch.distributed as dist
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(local_rank)
    else:
        local_rank = None

    if dp:
        from torch.nn import DataParallel



    args = Namespace(
        # yaml_path = "/home/hyohyeongjang/2024SWELL/codes/v2_config.yaml",
        # yaml_path = "/home/hyohyeongjang/2024SWELL/codes/v3_config.yaml",
        yaml_path = "/home/hyohyeongjang/2024SWELL/codes/v4_config.yaml",
        dp = dp,
        ddp = ddp,
        local_rank = local_rank
    )
    
    with open(args.yaml_path, 'r') as file:
        yaml_config = yaml.safe_load(file)
    
    #dictionary type string to dictionary
    yaml_config['dataStats']['labelToId'] = eval(yaml_config['dataStats']['labelToId'])
    yaml_config['dataStats']['labelToId'] = {"'"+"'" if key == "'" else key : value for key, value in yaml_config['dataStats']['labelToId'].items()}
    yaml_config['dataStats']['idToLabel'] = {value: key for key, value in yaml_config['dataStats']['labelToId'].items()}
    print("let's go!")
    main(args, yaml_config)
    