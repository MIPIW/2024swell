from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datetime import datetime
from tqdm import tqdm
from collections import Counter
from knockknock import slack_sender
from datasets import Dataset

webhook_url = "https://hooks.slack.com/services/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"
@slack_sender(webhook_url=webhook_url, channel="mine")
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

def main():

        # Define the checkpoint for RoBERTa base
    checkpoint = "FacebookAI/roberta-base"

    # Load a large corpus dataset for pretraining (e.g., Wikipedia, OpenWebText)
    # Replace 'wikitext' with your large corpus if available
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForMaskedLM.from_pretrained(checkpoint)

    trainer = set_trainer(dataset, model, tokenizer)
    trainer.train()




if __name__ == "__main__":
    print(f"started at {datetime.strftime(datetime.now(), '%Y.%m.%d. %H:%M:%S')}")

    main()

    print(f"ended at {datetime.strftime(datetime.now(), '%Y.%m.%d. %H:%M:%S')}")
