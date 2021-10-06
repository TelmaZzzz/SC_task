import os
import argparse
from utils import *
import torch
import datetime
from transformers import BertModel, BertTokenizer
from model import BERT
from train import *
import logging
import numpy as np
import csv
logging.getLogger().setLevel(logging.INFO)


class Example(object):
    def __init__(self, content, character, emotions=0, emotions_type=0, id=None):
        self.content = content
        self.character = character
        self.emotions = emotions
        self.list = ["小红","小亮","小明","小刚","小刘","小张","小李","小赵","小王","小白"]
        self.emotions_type = emotions_type
        self.id = id
    def rebuild(self):
        mp = dict()
        cnt = 0
        self.new_content = ""
        for idx, item in enumerate(self.content):
            self.new_content += item
            if item.isdigit():
                if self.content[idx - 1].islower():
                    ch = self.content[idx-1]+item
                    if mp.get(ch, None) is None:
                        mp[ch] = cnt
                        cnt += 1
                    self.new_content = self.new_content[:-2] + self.list[mp[ch]]
        self.new_character = "此时"+self.list[mp.get(self.character, 0)]+"的表情是"
        if self.emotions_type == 0:
            self.new_character += "充满爱意的。"
        elif self.emotions_type == 1:
            self.new_character += "高兴的。"
        elif self.emotions_type == 2:
            self.new_character += "惊讶的。"
        elif self.emotions_type == 3:
            self.new_character += "生气的。"
        elif self.emotions_type == 4:
            self.new_character += "恐惧的。"
        elif self.emotions_type == 5:
            self.new_character += "伤心的。"
        self.emotions = self.emotions / 3

    def out(self):
        debug("content", self.content)
        debug("character", self.character)
        debug("emotions", self.emotions)


def remake_data(data):
    mp = dict()
    for item in data:
        if mp.get(item.emotions, None) is None:
            mp[item.emotions] = [item]
        else:
            mp[item.emotions].append(item)
    num = 0
    for k, v in mp.items():
        if k != 0:
            num += len(v)
    num = num
    debug("num", num)
    res_data = []
    for k, v in mp.items():
        debug("k", k)
        debug("v", len(v))
        if k != 0:
            res_data += v
        else:
            res_data += v[:num]
    return res_data


def read_file(path, args):
    tsv = read_from_tsv(path)
    Examples = []
    for item in zip(tsv["content"], tsv["character"], tsv["emotions"]):
        # debug("emotions", item[2])
        if isinstance(item[1], float) or isinstance(item[2], float):
            continue
        # try:
        Examples.append(Example(item[0], item[1], int(item[2].split(",")[args.emotions_type]), args.emotions_type))
        Examples[-1].rebuild()
        # except Exception as e:
            # debug("item", item)
            # logging.warn(e)
    return Examples


def main(args):
    logging.info("Start prepare data...")
    train_data = read_file(args.train_path, args)
    valid_data = train_data[int(len(train_data)*0.95):]
    train_data = train_data[:int(len(train_data)*0.95)]
    train_data = remake_data(train_data)
    # _ = remake_data(train_data)
    tokenizer = BertTokenizer.from_pretrained(args.pre_train_name)
    train_sen = torch.LongTensor([bert_concat_tokenizer(item.new_content, item.new_character, tokenizer, args.fix_length)[0] for item in train_data])
    train_pad = torch.LongTensor([bert_concat_tokenizer(item.new_content, item.new_character, tokenizer, args.fix_length)[1] for item in train_data])
    train_label = torch.FloatTensor([item.emotions for item in train_data])
    valid_sen = torch.LongTensor([bert_concat_tokenizer(item.new_content, item.new_character, tokenizer, args.fix_length)[0] for item in valid_data])
    valid_pad = torch.LongTensor([bert_concat_tokenizer(item.new_content, item.new_character, tokenizer, args.fix_length)[1] for item in valid_data])
    valid_label = torch.FloatTensor([item.emotions for item in valid_data])
    train_dataset = torch.utils.data.TensorDataset(train_sen, train_pad, train_label)
    valid_dataset = torch.utils.data.TensorDataset(valid_sen, valid_pad, valid_label)
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size)
    model = BERT(args).cuda()
    logging.info("len: {}".format(len(train_iter)))
    train(train_iter, valid_iter, model, args)


def predict_main(args):
    test_data = read_from_tsv(args.test_path)
    Examples = []
    ans_mp = dict()
    ans_list = []
    cnt = 0
    predict_list = []
    logging.info("prepare test data...")
    for item in zip(test_data["id"], test_data["content"], test_data["character"]):
        ans_list.append(item[0])
        ans_mp[item[0]] = []
        if isinstance(item[2], float):
            ans_mp[item[0]] = [0,0,0,0,0,0]
            continue
        Examples.append(Example(item[1], item[2], id=cnt))
        cnt += 1
        predict_list.append(item[0])
        Examples[-1].rebuild()
    tokenizer = BertTokenizer.from_pretrained(args.pre_train_name)
    test_sen = torch.LongTensor([bert_concat_tokenizer(item.new_content, item.new_character, tokenizer, args.fix_length)[0] for item in Examples])
    test_pad = torch.LongTensor([bert_concat_tokenizer(item.new_content, item.new_character, tokenizer, args.fix_length)[1] for item in Examples])
    train_id = torch.LongTensor([item.id for item in Examples])
    test_dataset = torch.utils.data.TensorDataset(test_sen, test_pad, train_id)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
    model_list = [args.model_0, args.model_1, args.model_2, args.model_3, args.model_4, args.model_5]
    for idx, model_path in enumerate(model_list):
        logging.info("start predict model_{}".format(idx))
        model = torch.load(model_path).cuda()
        res = predict(test_iter, model, args)
        for item in res:
            if idx == 0:
                ans_mp[predict_list[int(item[1])]].append(0)
            else:
                ans_mp[predict_list[int(item[1])]].append(item[0])
    with open(args.predict_save_path, "w", newline="") as f:
        tsv_w = csv.writer(f, delimiter="\t")
        tsv_w.writerow(["id", "emotion"])
        for id in ans_list:
            tsv_w.writerow([id, list2str(ans_mp[id])])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="/Users10/lyzhang/Datasets/train_dataset_v2.tsv")
    parser.add_argument("--test_path", type=str, default="/Users10/lyzhang/Datasets/train_dataset_v2.tsv")
    parser.add_argument("--pre_train_name", type=str, default="bert-base-chinese")
    parser.add_argument("--emotions_type", type=int, default=0)
    parser.add_argument("--learning_rate", type=float,default=0.0003)
    parser.add_argument("--fc1_dim", type=int, default=256)
    parser.add_argument("--fc2_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--fix_length", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_step", type=int, default=100)
    parser.add_argument("--model_save_dir", type=str, default="/Users10/lyzhang/opt/tiger/SC_task/model")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--predict_save_path", type=str)
    parser.add_argument("--model_0", type=str)
    parser.add_argument("--model_1", type=str)
    parser.add_argument("--model_2", type=str)
    parser.add_argument("--model_3", type=str)
    parser.add_argument("--model_4", type=str)
    parser.add_argument("--model_5", type=str)

    args = parser.parse_args()
    if args.predict:
        predict_main(args)
    else:
        args.model_save_path = "/".join([args.model_save_dir, str(args.emotions_type) + "_" + d2s(datetime.datetime.now(), time=True)])
        main(args)

