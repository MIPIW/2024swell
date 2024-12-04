from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import os, glob
from functools import partial
from argparse import Namespace
import pickle

os.environ["TRUST_REMOTE_CODE"] = "True"

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

def set_trainer(args, dataset, tokenizer, model, task, word, val, ratio, preFunc = _preprocess_data):
# Preprocess dataset
    
    # Apply preprocessing
    preFunc = partial(preFunc, tokenizer = tokenizer, task = task)
    tokenized_datasets = dataset.map(preFunc, batched=True, num_proc = args.num_cores)

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Define metrics
    def compute_metrics(predictions):
        preds = np.argmax(predictions.predictions, axis=2)
        labels = predictions.label_ids
        preds = preds[labels != -100]
        labels = labels[labels != -100]
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
        return {"precision": precision, "recall": recall, "f1": f1}

    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.checkpoint_FTModel.format(task, word, val, ratio),
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        eval_steps=14,
        save_steps=14,
        logging_steps=14,
        learning_rate=2e-5,
        per_device_train_batch_size=512,
        per_device_eval_batch_size=512,
        load_best_model_at_end=True,
        save_total_limit=1,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train the model
    return trainer, tokenized_datasets



def set_trainer_testSingleToken(args, tokenized_datasets, tokenizer, model, task, lookup_token):

    lookup_token_id = tokenizer.convert_tokens_to_ids(lookup_token)
    print(lookup_token, lookup_token_id)

    def collate_fn(batch):
        input_ids = torch.stack([torch.tensor(x['input_ids']) for x in batch])
        attention_mask = torch.stack([torch.tensor(x['attention_mask']) for x in batch])
        labels = torch.tensor([x['labels'] for x in batch])
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

    batch_size = 512
    train_loader = DataLoader(tokenized_dataset['test'], batch_size=batch_size, collate_fn=collate_fn)

    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(args.device) for key, value in batch.items()}

            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    return correct / total

def main(args):
    
    if os.path.isfile(args.raw_FT):
        with open(args.raw_FT, "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = load_dataset("conll2003")  # replace with "universal_dependencies" for POS tagging
        with open(args.raw_FT, "wb") as f:
            pickle.dump(dataset, f)
    
    print(dataset)

    for task in args.tasks:
        label_list = dataset["train"].features[task].feature.names  # or "pos_tags" for POS tagging

        for val in ['count', 'ppl']:
            for word in args.words:
                for ratio in args.ratio:

                    if word in ['they', 'was'] and val == 'count' and task == "pos_tags":
                        continue

                    checkpoint = args.checkpoint_CTModel.format(word, val, ratio)
                    checkpoint = glob.glob(checkpoint + os.sep + 'checkpoint-*')[0]
                    print("\n", checkpoint, "\n")

                    model = AutoModelForTokenClassification.from_pretrained(checkpoint, num_labels=len(label_list))
                    
                    if args.DP:
                        model = DataParallel(model)
                    if args.DDP:
                        model.to(args.local_rank)
                        model = dist(model, device_ids=[args.local_rank])

                    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_baseModel, add_prefix_space = True)
                    trainer, tokenized_dataset = set_trainer(args, dataset, tokenizer, model, task, word, val, ratio)
                    trainer.train()
    


if __name__ == "__main__":
    
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
    )

    args.words = words
    args.local_rank = local_rank
    args.DP = DP
    args.DDP = DDP

    main(args)