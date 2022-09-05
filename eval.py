import numpy as np
import pandas as pd
from datasets import load_metric, Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
from colorama import Fore, Style

metric = load_metric("accuracy")
tokenizer = AutoTokenizer.from_pretrained("EMBEDDIA/sloberta")


def load_dataset(path="data/test.json"):
    return Dataset.from_pandas(pd.read_json(path_or_buf=path, lines=True))


data_test = load_dataset("data/test.jsonl")
data_val = load_dataset("data/val.jsonl")


def top_n_acc(n, model, p=False, v=True):
    score1 = 0
    classifier = pipeline("fill-mask", model=model, tokenizer=tokenizer, top_k=n)
    count = 0
    if v:
        for c in data_val:
            cl = classifier(c["sentence"])
            count += 1
            for j in range(n):
                if p:
                    if cl[j]["token_str"].casefold() == c["obj_label"].casefold():
                        score1 += 1
                        print(f" {Fore.GREEN} {cl[j]['sequence']} {Style.RESET_ALL}")
                    else:
                        print(f" {Fore.RED} {cl[j]['sequence']} {Style.RESET_ALL}")
                else:
                    if cl[j]["token_str"].casefold() == c["obj_label"].casefold():
                        score1 += 1
    else:
        for c in data_test:
            cl = classifier(c["sentence"])
            count += 1
            for j in range(n):
                if p:
                    if cl[j]["token_str"].casefold() == c["obj_label"].casefold():
                        score1 += 1
                        print(f" {Fore.GREEN} {cl[j]['sequence']} {Style.RESET_ALL}")
                    else:
                        print(f" {Fore.RED} {cl[j]['sequence']} {Style.RESET_ALL}")
                else:
                    if cl[j]["token_str"].casefold() == c["obj_label"].casefold():
                        score1 += 1

    return score1/count

def top_n_err_neg(n, model, p=False, v=True):
    score1 = 0
    classifier = pipeline("fill-mask", model=model, tokenizer=tokenizer, top_k=n)
    count = 0
    if v:
        for c in data_val:
            count += 1
            cl = classifier(c["sentence_negated"])
            for j in range(n):
                if p:
                    if cl[j]["token_str"].casefold() == c["obj_label_negated"].casefold():
                        score1 += 1
                        print(f" {Fore.RED} {cl[j]['sequence']} {Style.RESET_ALL}")
                    else:
                        print(f" {Fore.GREEN} {cl[j]['sequence']} {Style.RESET_ALL}")
                else:
                    if cl[j]["token_str"].casefold() == c["obj_label_negated"].casefold():
                        score1 += 1
    else:
        for c in data_test:
            count += 1
            cl = classifier(c["sentence_negated"])
            for j in range(n):
                if p:
                    if cl[j]["token_str"].casefold() == c["obj_label_negated"].casefold():
                        score1 += 1
                        print(f" {Fore.RED} {cl[j]['sequence']} {Style.RESET_ALL}")
                    else:
                        print(f" {Fore.GREEN} {cl[j]['sequence']} {Style.RESET_ALL}")
                else:
                    if cl[j]["token_str"].casefold() == c["obj_label_negated"].casefold():
                        score1 += 1
    return score1/count


def run(model, p=[False, False, False, False]):
    top1a = top_n_acc(1, model=model, p=p[0])
    top1e = top_n_err_neg(1, model=model, p=p[1])
    top5a = top_n_acc(5, model=model, p=p[0])
    top5e = top_n_err_neg(5, model=model, p=p[1])

    print("top 1 accuracy = {0:.1%}".format(top1a))
    print("top 1 error negated = {0:.1%}".format(top1e))
    print("top 5 accuracy = {0:.1%}".format(top5a))
    print("top 5 error negated = {0:.1%}".format(top5e))
    return top1a, top5a, top1e, top5e

    
if __name__ == "__main__":

    model = AutoModelForMaskedLM.from_pretrained("model-dip-base")
    v = False
    top1a = top_n_acc(1, model=model, p=False, v=v)
    top1e = top_n_err_neg(1, model=model, p=False, v=v)
    top5a = top_n_acc(5, model=model, p=False, v=v)
    top5e = top_n_err_neg(5, model=model, p=False, v=v)
    print("top 1 accuracy = {0:.1%}".format(top1a))
    print("top 1 error negated = {0:.1%}".format(top1e))
    print("top 5 accuracy = {0:.1%}".format(top5a))
    print("top 5 error negated = {0:.1%}".format(top5e))



