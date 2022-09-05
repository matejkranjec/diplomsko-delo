from transformers import AutoModelForMaskedLM, Trainer
from torch import nn
import torch

class KLTrainer(Trainer):
    def load_kl_model(self):
        print("copying model")
        self.kl = AutoModelForMaskedLM.from_pretrained("models/model-dip-base", output_loading_info=False, local_files_only=True)

        print("model copied")

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.get("logits")
        outputs2 = self.kl(**inputs)
        logits2 = outputs2.get("logits")
        shape = logits2.size()
        loss = 0
        for i in range(shape[0]):
            for j in range(shape[1]):
                sm1 = nn.functional.softmax(logits[i,j], dim=0)
                sm2 = nn.functional.softmax(logits2[i,j], dim=0)
                loss += torch.sum(sm2 * torch.log(sm2/sm1))
        loss /= shape[0]*shape[1]
        return (loss, outputs) if return_outputs else loss
