from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datetime import datetime
from tqdm import tqdm
from collections import Counter
from knockknock import slack_sender
from datasets import Dataset
from argparse import Namespace
import pickle 
import torch 
import sys, os
sys.path.append(os.path.expanduser('~/2024swell/dependencies/flash-roberta'))
# no flash_attn module
# from modeling_flash_roberta import FlashRobertaForMaskedLM

def set_trainer(args, dataset, model, tokenizer, word, ratio, val):

    # Tokenize dataset
    def tokenize_function(example):
        return tokenizer(example['text'], return_tensors="pt", truncation=True, padding="max_length", max_length=args.max_seq_len)

    # Tokenize dataset using map
    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=args.num_cores_train, remove_columns=["text"])
    tokenized_datasets.set_format(type = "torch")
    
    if args.DDP:
        model.to(args.local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=8, sampler=train_sampler)
    
    if args.DP:
        model = DataParallel(model)
    # tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=30, remove_columns=["text"])

    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # Define pretraining arguments
    training_args = TrainingArguments(
        output_dir=args.checkpoint_CTModel.format(word, val, ratio),
        overwrite_output_dir=True,
        num_train_epochs=5,  # Adjust for longer pretraining
        per_device_train_batch_size=args.batch_size,  # Adjust for available hardware
        save_strategy = "epoch",
        save_total_limit = 1,
        logging_strategy = "epoch",
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

webhook_url = "https://hooks.slack.com/services/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"
@slack_sender(webhook_url=webhook_url, channel="mine")
def main(args):

    for val in ['count', 'ppl']:
        for word in args.words:
            for ratio in args.ratio:
                print(word, ratio)

                print("loading datasets")
                with open(args.data_CT.format(word, val, ratio), "rb") as f:
                    exist, nonExist, ppl = pickle.load(f)
                
                exist = exist.map(lambda x: x.lower()).to_list()
                nonExist = nonExist.map(lambda x: x.lower()).to_list()
                dataset = exist + nonExist
                dataset_merged = Dataset.from_dict({'text': dataset})


                print("loading model")
                tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_baseModel)
                model = AutoModelForMaskedLM.from_pretrained(args.checkpoint_baseModel, torch_dtype=torch.bfloat16)
                # model = FlashRobertaForMaskedLM.from_pretrained(args.checkpoint_baseModel, torch_dtype=torch.bfloat16)
                # replace embedding of the word with arbitrary tensor 
                # default: False
                if args.do_RandomInitialize:
                    print("change initial embeddings")
                    embedding_layer = model.get_input_embeddings()
                    token_id = tokenizer.convert_tokens_to_ids(word)
                    original_embedding = embedding_layer.weight[token_id]
                    print("embedding changed from ", original_embedding[:10])
                    original_mean, original_std = original_embedding.mean(), original_embedding.std()

                    arbitrary_tensor = torch.rand_like(original_embedding)
                    arbitrary_mean, arbitrary_std = arbitrary_tensor.mean(), arbitrary_tensor.std()
                    arbitrary_tensor = (arbitrary_tensor - arbitrary_mean) / arbitrary_std
                    arbitrary_tensor = (arbitrary_tensor * original_std) + original_mean

                    with torch.no_grad():
                        embedding_layer.weight[token_id] = arbitrary_tensor
                
                    embedding_layer = model.get_input_embeddings()
                    print("to", embedding_layer.weight[token_id][:10])

                trainer = set_trainer(args, dataset_merged, model, tokenizer, word, ratio, val)
                trainer.train()
            


if __name__ == "__main__":
    
    # should not be both True
    # could be both False
    DDP = False
    DP = False
    global_rank = '0'
    if DP or DDP:
        os.environ["CUDA_VISIBLE_DEVICES"] = global_rank
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    if DDP:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        local_rank = None

    if DP:
        from torch.nn import DataParallel

        


    print(f"started at {datetime.strftime(datetime.now(), '%Y.%m.%d. %H:%M:%S')}")

    # words = ['and', 'one', 'the', 'for', 'new', 'time', 'they', 'was', 'has', 'that', 'who', 'when']
    words = ['one', 'for', 'new', 'time', 'they', 'was', 'has', 'that', 'who']
    words = ['they', 'was', 'has', 'that', 'who']
    
    # words = ['and', 'the', 'time', 'was', 'who']

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
        batch_size = 64, # 64에 약 25000MiB(100%)
        do_RandomInitialize = False,
        num_cores_train = 10

    )

    args.words = words
    args.local_rank = local_rank
    args.DP = DP
    args.DDP = DDP

    main(args)

    print(f"ended at {datetime.strftime(datetime.now(), '%Y.%m.%d. %H:%M:%S')}")
