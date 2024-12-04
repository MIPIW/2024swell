from datetime import datetime
from knockknock import slack_sender
from datasets import load_dataset
from collections import Counter
from transformers import AutoTokenizer
from tqdm import tqdm

webhook_url = "https://hooks.slack.com/services/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"
@slack_sender(webhook_url=webhook_url, channel="mine")
def main():
    # dataset = load_dataset("conll2003")  # replace with "universal_dependencies" for POS tagging
    # label_list = dataset["train"].features["pos_tags"].feature.names  # or "pos_tags" for POS tagging
    
    # # x = []
    # # for example in dataset['validation']['tokens']:
    # #     if "for" in example:
    # #         x.append([(i, "for") for i in example])

    # print(len(Counter([tuples for example in x for tuples in example])))
    # print(len(dataset['validation']['tokens']))

    # print(dataset['validation']['pos_tags'])


    return 
if __name__ == "__main__":
    print(f"started at {datetime.strftime(datetime.now(), '%Y.%m.%d. %H:%M:%S')}")

    main()

    print(f"ended at {datetime.strftime(datetime.now(), '%Y.%m.%d. %H:%M:%S')}")
