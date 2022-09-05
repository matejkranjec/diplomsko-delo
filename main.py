from cProfile import label
import pandas as pd
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer, default_data_collator, TrainingArguments, Trainer, is_datasets_available
from datasets import Dataset
import collections
import functools
from trainers.KLTrainer import KLTrainer
from trainers.ULTrainer import ULTrainer
import eval
import visualize

tokenizer = AutoTokenizer.from_pretrained("EMBEDDIA/sloberta")
model = AutoModelForMaskedLM.from_pretrained("EMBEDDIA/sloberta", output_loading_info=False)



def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

def load_dataset(path="data/train.jsonl"):
    data = Dataset.from_pandas(pd.read_json(path_or_buf=path, lines=True))
    examples = []
    for d in data:
        examples.append(d["sentence"].replace("<mask>",d["obj_label"]))
    data = data.add_column("text", examples)
    tokenized_data = data.map(tokenize_function, batched=True)
    tokenized_data = tokenized_data.add_column("labels",tokenized_data["input_ids"])
    return tokenized_data


def load_dataset_ul(path="data/train.jsonl"):
    data = Dataset.from_pandas(pd.read_json(path_or_buf=path, lines=True))
    examples = []
    for d in data:
        examples.append(d["sentence_negated"].replace("<mask>",d["obj_label_negated"])+" "+ d["sentence"].replace("<mask>",d["obj_label"]))
    data = data.add_column("text", examples)
    tokenized_data = data.map(tokenize_function, batched=True)
    tokenized_data = tokenized_data.add_column("labels",tokenized_data["input_ids"])
    tokenized_data = tokenized_data.remove_columns("obj_label")
    tokenized_data = tokenized_data.rename_columns({"obj_label_negated":"obj_label"})
    return tokenized_data

def load_dataset_kl(path="data/train.jsonl"):
    data = Dataset.from_pandas(pd.read_json(path_or_buf=path, lines=True))
    examples = []
    for d in data:
        examples.append(d["sentence"].replace("<mask>",d["obj_label"])+" "+ d["sentence"].replace("<mask>",d["obj_label"]))
    data = data.add_column("text", examples)
    tokenized_data = data.map(tokenize_function, batched=True)
    tokenized_data = tokenized_data.add_column("labels",tokenized_data["input_ids"])
    return tokenized_data


def group_texts(examples):
    labels = examples.pop("obj_label")
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    c = np.array(concatenated_examples["input_ids"])
    mask = [0 for _ in concatenated_examples["input_ids"]]
    labels = tokenizer(labels)["input_ids"]
    for i in range(len(labels)):
        _, *labels[i], _ = labels[i]

    temp = 0
    for label in labels:
        length = len(label)
        if length == 1:
            index = np.where(c[temp:] == label[0])[0][0]
            mask[temp+index] = 1
            temp += index
        else:
            n = [c[temp+i:temp+i+length] for i in range(len(c[temp:])+1-length)]
            for idx, arr in enumerate(n):
                if(cmp_list(arr,label)):
                    for l in range(length):
                        mask[temp+idx+l] = 1
                    temp += idx+length
                    break
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // chunk_size) * chunk_size
    result = {
        k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    m = [mask[i:i+chunk_size] for i in range(0, total_length, chunk_size)]
    result["mask"] = m
    return result

def cmp_list(l1,l2):
    return functools.reduce(lambda x, y : x and y, map(lambda p, q: p == q,l1,l2), True)

def wwm_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")
        mask = feature.pop("mask")
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
    return default_data_collator(features)


def insert_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = wwm_data_collator(features)
    return {"masked_" + k : v.numpy() for k, v in masked_inputs.items()}

if __name__ == "__main__":
    data = load_dataset()
    data_kl = load_dataset_kl()
    data_ul = load_dataset_ul()
    
    data = data.remove_columns(["obj_label_negated", "sentence", "sentence_negated", "text"])
    data_kl = data_kl.remove_columns(["obj_label_negated", "sentence", "sentence_negated", "text"])
    data_ul = data_ul.remove_columns(["sentence", "sentence_negated", "text"])
    chunk_size = 32
    data = data.map(group_texts, batched=True)
    data_ul = data_ul.map(group_texts, batched=True)
    data_kl = data_kl.map(group_texts, batched=True)


    batch_size = 8
    training_args = TrainingArguments(
        num_train_epochs=10,
        remove_unused_columns=False,
        output_dir = "models/model-dip-base",
        overwrite_output_dir = True,
        evaluation_strategy = "no",
        learning_rate = 1e-4,
        weight_decay = 0.01,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        logging_steps = len(data)//batch_size,
        save_strategy = "no",
    )

    training_args_kl = TrainingArguments(
        num_train_epochs=1,
        remove_unused_columns=False,
        output_dir = "trainers/kl",
        overwrite_output_dir = True,
        evaluation_strategy = "no",
        learning_rate = 2e-5,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        logging_steps = len(data_kl)//batch_size,
        save_strategy = "no",
        lr_scheduler_type = "constant"
    )
    training_args_neg = TrainingArguments(
        num_train_epochs=1,
        remove_unused_columns=False,
        output_dir = "trainers/neg",
        overwrite_output_dir = True,
        evaluation_strategy = "no",
        learning_rate = 2e-5,
        #weight_decay = 0.01,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        logging_steps = len(data_ul)//batch_size,
        save_strategy = "no",
        lr_scheduler_type = "constant"
    )

    data = data.map(insert_mask, batched=True, remove_columns=data.column_names)
    data_kl = data_kl.map(insert_mask, batched=True, remove_columns=data_kl.column_names)
    data_ul = data_ul.map(insert_mask, batched=True, remove_columns=data_ul.column_names)

    data = data.rename_columns({"masked_input_ids": "input_ids", "masked_attention_mask":"attention_mask","masked_labels":"labels"})
    data_kl = data_kl.rename_columns({"masked_input_ids": "input_ids", "masked_attention_mask":"attention_mask","masked_labels":"labels"})
    data_ul = data_ul.rename_columns({"masked_input_ids": "input_ids", "masked_attention_mask":"attention_mask","masked_labels":"labels"})
    #trainer = Trainer(model=model,args=training_args,train_dataset=data, eval_dataset=data, data_collator=default_data_collator)
    #trainer.train()
    #model.save_pretrained(save_directory="models/model-dip-base")
    model = AutoModelForMaskedLM.from_pretrained("models/model-dip-base", output_loading_info=False, local_files_only=True)
    trainer_neg = ULTrainer(model=model,args=training_args_neg,train_dataset=data_ul, eval_dataset=data_ul, data_collator=default_data_collator)
    trainer_kl = KLTrainer(model=model,args=training_args_kl,train_dataset=data_kl, eval_dataset=data_kl, data_collator=default_data_collator)

    trainer_kl.load_kl_model()

    print("evaluating base model")
    r = eval.run(model)
    top1a = []
    top1e = []
    top5a = []
    top5e = []
    top1a.append(r[0])
    top1e.append(r[2])
    top5a.append(r[1])
    top5e.append(r[3])

    for iter in range(20):
        trainer_neg.train()
        trainer_kl.train()
        print(f"evaluating epoch {iter+1}")
        r = eval.run(model)
        top1a.append(r[0])
        top1e.append(r[2])
        top5a.append(r[1])
        top5e.append(r[3])
        print("top 1 accuracies ",top1a)
        print("top 1 neg errors ",top1e)
        print("top 5 accuracies ",top5a)
        print("top 5 neg errors ",top5e)
        model.save_pretrained(save_directory = f"models/model-dip-{iter+1}")
    visualize.plot(top1a, top1e, top5a, top5e)

