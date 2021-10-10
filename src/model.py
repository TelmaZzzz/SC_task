import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from utils import *

class BERT(nn.Module):
    def _init_normal(self, m):
        if type(m) == nn.Linear:
            nn.init.uniform_(m.weight, a=-0.0001, b=0.0001)
            nn.init.zeros_(m.bias)

    def __init__(self, args):
        super(BERT, self).__init__()
        # with torch.no_grad():
        self._bert = BertModel.from_pretrained(args.pre_train_name)
        # for param in self._bert.parameters():
        #     param.requires_grad = False
        #     debug("bert", param)
        self._FC = nn.Sequential(
            nn.Linear(768, args.fc1_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.fc1_dim, args.fc2_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.fc2_dim, 6)
        )
        self._FC.apply(self._init_normal)
        self._F = nn.Linear(768, 6)
        self._F.apply(self._init_normal)

    def forward(self, sen, pad):
        # output = self._FC(self._bert(sen)[1])
        # output = self._FC(self._bert(sen, return_dict=False)[1])
        x = self._bert(input_ids=sen, attention_mask=pad, return_dict=False)[1]
        # debug("x", x.size())
        # debug("x", x)
        output = self._FC(x)
        # output = self._F(x)
        # debug("output", output)
        # output = self._FC(self._bert(sen, return_dict=False)[1])
        return output