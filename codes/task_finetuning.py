from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import os
from functools import partial

os.environ["TRUST_REMOTE_CODE"] = "True"


def set_trainer(dataset, preprocess_function, tokenizer, model, task, out_path):
# Preprocess dataset
    
    # Apply preprocessing
    preprocess_function = partial(preprocess_function, tokenizer = tokenizer, task = task)
    tokenized_datasets = dataset.map(preprocess_function, batched=True, num_proc = 30)

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
        output_dir=f"/home/hyohyeongjang/2024aut_comprac/weights/{out_path}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=512,
        per_device_eval_batch_size=512,
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



def set_trainer_testOnlyFor(tokenized_datasets, tokenizer, model, task, out_path, lookup_token):
    
    lookup_token_id = tokenizer.convert_tokens_to_ids(lookup_token)

    # Define metrics
    def compute_metrics(predictions):
        preds = np.argmax(predictions.predictions, axis=2)
        labels = predictions.label_ids
        preds = preds[labels == lookup_token_id]
        labels = labels[labels == lookup_token_id]
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
        return {"precision": precision, "recall": recall, "f1": f1}

    
    # Training arguments
    training_args = TrainingArguments( # training process will not be conducted
        output_dir=f"/home/hyohyeongjang/2024aut_comprac/weights/{out_path}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=512,
        per_device_eval_batch_size=512,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"], # not used
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    return trainer

def main():
    task = "pos_tags"
    lookup_token = "for1"
    # Load dataset for POS or NER (e.g., Universal Dependencies for POS, or conll2003 for NER)
    dataset = load_dataset("conll2003")  # replace with "universal_dependencies" for POS tagging
    label_list = dataset["train"].features[task].feature.names  # or "pos_tags" for POS tagging
    tokenizer_checkpoint = "FacebookAI/roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint, add_prefix_space=True)
        
    # Load RoBERTa tokenizer and model
    model_checkpoint = "/home/hyohyeongjang/2024aut_comprac/weights/roberta-pretrained/checkpoint-39902"
    out_path = "roberta-finetuneds-full"
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

    trainer, tokenized_dataset = set_trainer(dataset, _preprocess_data, tokenizer, model, task, out_path)
    # trainer.train()

    for i in [28,56,84]:
        eval_checkpoint = f"/home/hyohyeongjang/2024aut_comprac/weights/roberta-finetuneds-full/checkpoint-{i}"
        model = AutoModelForTokenClassification.from_pretrained(eval_checkpoint, num_labels=len(label_list))
        eval_trainer = set_trainer_testOnlyFor(tokenized_dataset, tokenizer, model, task, out_path, lookup_token)   
        print(eval_trainer.evaluate())
        

    model_checkpoint = "/home/hyohyeongjang/2024aut_comprac/weights/roberta-pretrained-notFiltered/checkpoint-56293"
    out_path = "roberta-notFiltered-full"
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

    trainer, tokenized_dataset = set_trainer(dataset, _preprocess_data, tokenizer, model, task, out_path)
    # trainer.train()

    for i in [28,56,84]:
        eval_checkpoint = f"/home/hyohyeongjang/2024aut_comprac/weights/roberta-notFiltered-full/checkpoint-{i}"
        model = AutoModelForTokenClassification.from_pretrained(eval_checkpoint, num_labels=len(label_list))
        eval_trainer = set_trainer_testOnlyFor(tokenized_dataset, tokenizer, model, task, out_path, lookup_token )   
        print(eval_trainer.evaluate())
    
    

if __name__ == "__main__":
    


    main()