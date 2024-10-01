from torch.utils.data import Dataset
from torchtext.datasets import SST2
import torch
import math


class sst2Dataset(Dataset):
    def __init__(self, device, max_length=128,
                 isTrain=True, tokenizer=None,
                 batch_size=16, d_model=8):
        self.max_length = max_length
        self.isTrain = isTrain
        if self.isTrain:
            self.data_iter = SST2(split="train")
        else:
            self.data_iter = SST2(split="dev")
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.d_model = d_model

    def __len__(self):
        return len(list(self.data_iter))

    def collate_batch(self, batch):
        ids, types, masks, label_list = [], [], [], []
        vocab_size = len(self.tokenizer.vocab)
        for text, label in batch:
            tokenized = self.tokenizer(text,
                                       padding="max_length", max_length=self.max_length,
                                       truncation=True, return_tensors="pt")
            ids.append(tokenized['input_ids'])
            types.append(tokenized['token_type_ids'])
            masks.append(tokenized['attention_mask'])
            label_list.append(label)
        embedding_layer = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.d_model)
        # inputids = embedding_layer(torch.squeeze(torch.stack(ids)))
        inputids = torch.squeeze(torch.stack(ids))

        input_data = {
            "input_ids": inputids.to(self.device),
            "token_type_ids": torch.squeeze(torch.stack(types)).to(self.device),
            "attention_mask": torch.squeeze(torch.stack(masks)).to(self.device)
        }
        #
        if self.isTrain:
            return input_data, label_list
        else:
            # label_list = torch.tensor(label_list, dtype=torch.int64).to(self.device).float()
            return input_data, label_list