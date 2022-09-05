import torch
from torch import nn
from transformers import Trainer

class ULTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        s = labels.size()
        mask = torch.zeros(8,32)
        loss = 0
        for i in range(s[0]):
            for j in range(s[1]):
                if labels[i,j] != inputs.get("input_ids")[i,j]:
                    mask[i,j] = 1
        for i in range(s[0]):
            for j in range(s[1]):
                if mask[i,j] == 1:
                    loss -= torch.log(1-nn.functional.softmax(logits[i,j], dim=0)[labels[i,j]])
        loss /= torch.sum(mask)
        return (loss, outputs) if return_outputs else loss

