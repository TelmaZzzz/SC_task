import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, AutoModel
from utils import *


class LayerNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-12):
        super(LayerNorm, self).__init__()
        self._weight = nn.Parameter(torch.ones(hidden_dim))
        self._bias = nn.Parameter(torch.zeros(hidden_dim))
        self._eps = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self._eps)
        return self._weight * x + self._bias


class BERT(nn.Module):
    def _init_normal(self, m):
        if type(m) == nn.Linear:
            nn.init.uniform_(m.weight, a=-0.0001, b=0.0001)
            nn.init.zeros_(m.bias)

    def __init__(self, args):
        super(BERT, self).__init__()
        # with torch.no_grad():
        self._bert = AutoModel.from_pretrained(args.pre_train_name)
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

    def forward(self, ids, mask, token_type_ids):
        # output = self._FC(self._bert(sen)[1])
        # output = self._FC(self._bert(sen, return_dict=False)[1])
        x = self._bert(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)[0][:,0,:]
        # debug("x", x.size())
        # debug("x", x)
        output = self._FC(x)
        # output = self._F(x)
        # debug("output", output)
        # output = self._FC(self._bert(sen, return_dict=False)[1])
        return output


class BERT_LSTM(nn.Module):
    def _init_normal(self, m):
        if type(m) == nn.Linear:
            nn.init.uniform_(m.weight, a=-0.0001, b=0.0001)
            nn.init.zeros_(m.bias)
    
    def __init__(self, args):
        super(BERT_LSTM, self).__init__()
        self.bert = AutoModel.from_pretrained(args.pre_train_name)
        self.LSTM = nn.LSTM(768, args.hidden_size, batch_first=True, dropout=args.dropout, bidirectional=True)
        self.lstm_fix_len = args.fix_length_lstm
        self.FC = nn.Sequential(
            nn.Linear(args.hidden_size * 2 + 768, args.fc1_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.fc1_dim, args.fc2_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.fc2_dim, 6)
        )
        self.FC.apply(self._init_normal)
        self.LSTM.apply(self._init_normal)
    
    def _pooling(self, v):
        return torch.cat([F.avg_pool1d(v.transpose(1, 2), v.size(1)).squeeze(-1), F.max_pool1d(v.transpose(1, 2), v.size(1)).squeeze(-1)], -1)

    def forward(self, sen, sen_id, pad, start_end, character):
        x, _ = self.bert(input_ids=sen, attention_mask=pad, token_type_ids=sen_id, return_dict=False)
        CLS = x[:,0,:]
        batch, len, hidden_size = x.size()
        x = x.view(batch * len, -1)
        debug("x shape", x.shape)
        debug("pad shape", torch.zeros_like(x[0, :]).view(1, -1).shape)
        x = torch.cat([torch.zeros_like(x[0, :]).view(1, -1), x], dim=0)
        batch_start_end = []
        for idx, se in enumerate(start_end):
            debug("se", se)
            offset = idx * len + 1
            debug("offset", offset)
            batch_start_end.extend(range(offset + int(se[0]), offset + int(se[1])))
            num = self.lstm_fix_len - (int(se[1]) - int(se[0]))
            if num < 0:
                batch_start_end = batch_start_end[:num]
            else:
                batch_start_end.extend([0] * num)
        batch_start_end = torch.tensor(batch_start_end).view(-1).cuda()
        inputs = torch.index_select(x, 0, batch_start_end).reshape(batch, self.lstm_fix_len, hidden_size)
        debug("inputs shape", inputs.shape)
        y, _ = self.LSTM(inputs)
        batch, len, hidden_size = y.size()
        batch_character_1 = []
        batch_character_2 = []
        y = y.reshape(batch * len, hidden_size)
        for idx, se in enumerate(character):
            debug("se2", se)
            offset = idx * len
            if se[1] >= len:
                batch_character_1.append(offset + len // 2)
                batch_character_2.append(offset + len // 2 + 1)
            else:
                batch_character_1.append(offset + se[0])
                batch_character_2.append(offset + se[1])
        batch_character_1 = torch.tensor(batch_character_1).view(-1).cuda()
        batch_character_2 = torch.tensor(batch_character_2).view(-1).cuda()
        debug("batch_character_1", batch_character_1)
        debug("batch_character_2", batch_character_2)
        character_start = torch.index_select(y, 0, batch_character_1).reshape(batch, hidden_size)
        character_end = torch.index_select(y, 0, batch_character_2).reshape(batch, hidden_size)
        character_out = torch.cat([character_start[:, :hidden_size // 2], character_end[:, hidden_size // 2:]], -1)

        # y = y.view(batch * len, -1)
        # debug("y shape", y.shape)
        # y = self._pooling(y)
        # debug("y0 shape", y.shape)
        terminal = self.FC(torch.cat([CLS, character_out.view(batch, -1)], dim=-1))
        debug("terminal shape", terminal.shape)
        return terminal


class BERT_prompt(nn.Module):
    def _init_normal(self, m):
        if type(m) == nn.Linear:
            nn.init.uniform_(m.weight, a=-0.0001, b=0.0001)
            nn.init.zeros_(m.bias)

    def __init__(self, args):
        super(BERT_prompt, self).__init__()
        # with torch.no_grad():
        self._bert = AutoModel.from_pretrained(args.pre_train_name)
        # logging.info(self._bert)
        # for param in self._bert.parameters():
        #     param.requires_grad = False
        #     debug("bert", param)
        self._FC = nn.Sequential(
            nn.Linear(768 * 6, args.fc1_dim),
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

    def forward(self, ids, mask, token_type_ids):
        # output = self._FC(self._bert(sen)[1])
        # output = self._FC(self._bert(sen, return_dict=False)[1])
        batch, num, len = ids.size()
        ids = ids.reshape(batch * num, len)
        mask = mask.reshape(batch * num, len)
        token_type_ids = token_type_ids.reshape(batch * num, len)
        debug("ids shape", ids.shape)
        debug("maske shape", mask.shape)
        debug("token_type_ids shape", token_type_ids.shape)
        x = self._bert(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)[0][:,0,:]
        # debug("x", x.size())
        # debug("x", x)
        x = x.reshape(batch, -1)
        output = self._FC(x)
        # output = self._F(x)
        # debug("output", output)
        # output = self._FC(self._bert(sen, return_dict=False)[1])
        return output
