import collections
import os
import argparse
from utils import *
import torch
import datetime
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
from model import BERT, BERT_LSTM, BERT_prompt
from train import *
import logging
import numpy as np
import csv
import copy
import random
from sklearn.model_selection import KFold

logging.getLogger().setLevel(logging.INFO)
random.seed(19980917)
torch.manual_seed(19980917) #为CPU设置种子用于生成随机数，以使得结果是确定的
torch.cuda.manual_seed(19980917) #为当前GPU设置随机种子


class Example(object):
    def __init__(self, content, character, emotions=0, emotions_type=0, id=None):
        self.content = content
        self.character = character
        self.emotions = emotions
        self.list = ["小红","小亮","小明","小刚","小刘","小张","小李","小赵","小王","小白","小冯","小马","小叶"]
        self.emotions_type = emotions_type
        self.id = id
        self.up_content = ""

    def rebuild(self):
        mp = dict()
        cnt = 0
        self.new_content = self.content
        self.new_character = None
        flag = False
        zi = (0, 0)
        for idx, item in enumerate(self.content):
            if idx > 0:
                if self.content[idx-1].islower() and self.content[idx].isdigit():
                    zi = (idx - 1, idx)
                if self.content[idx-1:idx+1] == self.character:
                    self.new_character = (idx - 1, idx)
                    flag = True
        if flag is False:
            self.new_character = zi
        if self.new_character is None:
            logging.warn("content:{}. character:{}".format(self.content, self.character))
        prompt = "此时" + self.character + "的表情是"
        # word_list = ["充满爱意的。", "高兴的。", "惊讶的", "生气的", "害怕的", "难过的"]
        # self.prompt = []
        # for word in word_list:
        #     self.prompt.append(prompt + word)
        self.prompt = prompt + "？"
        cls = False
        if cls is False:
            # self.emotions = self.emotions / 3
            self.emotions = [item / 3 for item in self.emotions]

    def add_up(self, up_content):
        self.up_content = self.up_content + up_content
        # self.content = up_content + self.content
    
    def get_emotion(self):
        self.emotions = self.emotions.split(",")
        self.emotions = [int(item) for item in self.emotions]
        # self.emotions = int(self.emotions.split(",")[self.emotions_type])

    def out(self):
        debug("content", self.content)
        debug("character", self.character)
        debug("emotions", self.emotions)


class Story(object):
    def __init__(self, id):
        self.id = id
        self.content = dict()
        self.examples = dict()

    def add_example(self, id, example):
        self.examples[id] = example

    def rebuild(self):
        examples = sorted(self.examples.items(), key=lambda x:x[0])
        pre = ""
        cnt = 0
        mp = dict()
        for item in examples:
            if item[1].content != pre:
                self.content[cnt] = item[1].content
                mp[item[0]] = cnt
                cnt += 1
            mp[item[0]]=cnt-1
            pre = item[1].content
        Examples = []
        for item in examples:
            if item[1].character is None:
                continue
            if len(item[1].character) == 0:
                continue
            # debug("character", type(item[1].character))
            add_item = copy.deepcopy(item[1])
            pos = mp[item[0]]
            for i in range(1, 4):
                up_content = self.content.get(pos-i, "")
                add_item.add_up(up_content)
            add_item.get_emotion()
            add_item.rebuild()
            Examples.append(add_item)
            # debug("content", add_item.new_content)
        return Examples


class RerankingCollate:
    def __init__(self, MAX_LEN, predict=False):
        self.CONFIG = {}
        self.CONFIG['BUCKET'] = True
        self.CONFIG['MAX_LEN'] = MAX_LEN
        self.predict = predict

    def __call__(self, batch):
        out = {
                'ids'               : [], # torch.longに型変換
                'mask'              : [], # torch.longに型変換
                'token_type_ids'    : [], # torch.longに型変換
                'label': []
            }

        for i in range(len(batch)):
            for k, v in batch[i].items():
                out[k].append(v)

        # Deciding the number of padding
        if self.CONFIG['BUCKET']:
            max_pad = 0
            for p in out['ids']:
                if len(p) > max_pad:
                    max_pad = len(p)
            if max_pad > 512:
                logging.warn("MAX_PAD ERROR")
        else:
            max_pad = self.CONFIG['MAX_LEN']
            
        # Padding
        for i in range(len(batch)):
            tokenized_text = out['ids'][i]
            token_type_ids = out['token_type_ids'][i]
            mask           = out['mask'][i]
            text_len       = len(tokenized_text)
            # debug("text_len", text_len)
            # debug("max_pad", max_pad)
            out['ids'][i] = (tokenized_text + [0] *(max_pad - text_len))[: max_pad]
            out['token_type_ids'][i] = (token_type_ids + [1] * (max_pad - text_len))[: max_pad]
            out['mask'][i] = (mask + [0] * (max_pad - text_len))[: max_pad]
            if len(out["ids"][i]) > 512:
                logging.warn("ERROR")
                # debug("out ids i j", len(out["ids"][i][j]))
                # debug("out token_type_ids i j", len(out["token_type_ids"][i][j]))
                # debug("out mask i j", len(out["mask"][i][j]))
                
        # torch.float
        if self.predict:
            out['label'] = torch.tensor(out['label'], dtype=torch.long)
        else:
            out['label']          = torch.tensor(out['label'], dtype=torch.float)
        # torch.long
        out['ids'] = torch.tensor(out['ids'], dtype=torch.long)
        out['mask']           = torch.tensor(out['mask'], dtype=torch.long)
        out['token_type_ids'] = torch.tensor(out['token_type_ids'], dtype=torch.long)

        return out


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, Examples, tokenizer, args):
        self.label = []
        self.sen = []
        self.pad = []
        self.sen_id = []
        self.start_end = []
        self.character = []
        self.fix_length = args.fix_length
        if args.predict:
            self.build_predict(Examples, tokenizer)
        else:
            self.build(Examples, tokenizer)
    
    def __getitem__(self, idx):
        return self.sen[idx], self.sen_id[idx], self.pad[idx], self.start_end[idx], self.label[idx], self.character[idx]

    def __len__(self):
        return len(self.label)

    def build(self, Examples, tokenizer):
        for item in Examples:
            sen, pad, sen_id, start, end = bert_concat_id_tokenize(item.up_content, item.new_content, tokenizer, self.fix_length)
            self.sen.append(torch.LongTensor(sen))
            self.pad.append(torch.LongTensor(pad))
            self.sen_id.append(torch.LongTensor(sen_id))
            self.label.append(torch.FloatTensor(item.emotions))
            self.start_end.append(torch.LongTensor((start, end)))
            # debug("new_character", item.new_content)
            # debug("character", item.character)
            self.character.append(torch.LongTensor(item.new_character))

        debug("label len", len(self.label))
        debug("sen len", len(self.sen))
        assert len(self.sen) == len(self.pad)
        assert len(self.sen) == len(self.sen_id)
        assert len(self.sen) == len(self.label)
        assert len(self.sen) == len(self.start_end)
    
    def build_predict(self, Examples, tokenizer):
        for item in Examples:
            sen, pad, sen_id, start, end = bert_concat_id_tokenize(item.up_content, item.new_content, tokenizer, self.fix_length)
            self.sen.append(torch.LongTensor(sen))
            self.pad.append(torch.LongTensor(pad))
            self.sen_id.append(torch.LongTensor(sen_id))
            self.label.append(torch.LongTensor([item.id]).view(-1))
            self.start_end.append(torch.LongTensor((start, end)))
            # debug("new_character", item.new_content)
            # debug("character", item.character)
            self.character.append(torch.LongTensor(item.new_character))
        #     debug("sen", self.sen[-1].shape)
        #     debug("pad", self.pad[-1].shape)
        #     debug("sen_id", self.sen_id[-1].shape)
        #     debug("label", self.label[-1].shape)
        #     debug("start_end", self.start_end[-1].shape)
        #     debug("character", self.character[-1].shape)

        # debug("label len", len(self.label))
        # debug("sen len", len(self.sen))
        assert len(self.sen) == len(self.pad)
        assert len(self.sen) == len(self.sen_id)
        assert len(self.sen) == len(self.label)
        assert len(self.sen) == len(self.start_end)


class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, Examples, tokenizer, args):
        self.label = []
        self.ids = []
        self.mask = []
        self.token_type_ids = []
        self.start_end = []
        self.fix_length = args.fix_length
        if args.predict:
            self.build_predict(Examples, tokenizer)
        else:
            self.build(Examples, tokenizer)
    
    def __getitem__(self, idx):
        return {
            "ids": self.ids[idx],
            "mask": self.mask[idx],
            "token_type_ids": self.token_type_ids[idx],
            "label": self.label[idx]
        }

    def __len__(self):
        return len(self.label)

    def build(self, Examples, tokenizer):
        for item in Examples:
            ids, mask, token_type_ids = bert_concat_tokenizer_list(item.up_content, item.content, item.prompt, tokenizer, self.fix_length)
            self.ids.append(ids)
            self.mask.append(mask)
            self.token_type_ids.append(token_type_ids)
            self.label.append(item.emotions)

    def build_predict(self, Examples, tokenizer):
        for item in Examples:
            pass


class BERTDataset(torch.utils.data.Dataset):
    def __init__(self, Examples, tokenizer, args):
        self.label = []
        self.ids = []
        self.mask = []
        self.token_type_ids = []
        self.start_end = []
        self.fix_length = args.fix_length
        if args.predict:
            self.build_predict(Examples, tokenizer)
        else:
            self.build(Examples, tokenizer)
    
    def __getitem__(self, idx):
        return {
            "ids": self.ids[idx],
            "mask": self.mask[idx],
            "token_type_ids": self.token_type_ids[idx],
            "label": self.label[idx]
        }

    def __len__(self):
        return len(self.label)

    def build(self, Examples, tokenizer):
        for item in Examples:
            ids, mask, token_type_ids = bert_concat_tokenizer_new(item.content + item.prompt, item.up_content, tokenizer, self.fix_length)
            self.ids.append(ids)
            self.mask.append(mask)
            self.token_type_ids.append(token_type_ids)
            self.label.append(item.emotions)
            # debug("ids len", len(self.ids[-1]))

    def build_predict(self, Examples, tokenizer):
        for item in Examples:
            ids, mask, token_type_ids = bert_concat_tokenizer_new(item.content + item.prompt, item.up_content, tokenizer, self.fix_length)
            self.ids.append(ids)
            self.mask.append(mask)
            self.token_type_ids.append(token_type_ids)
            self.label.append(item.id)


def remake_data(data):
    mp = dict()
    for item in data:
        if mp.get(item.emotions, None) is None:
            mp[item.emotions] = [item]
        else:
            mp[item.emotions].append(item)
    num = 0
    for k, v in mp.items():
        if k == 0:
            num = len(v) // 4
    debug("num", num)
    res_data = []
    for k, v in mp.items():
        debug("k", k)
        debug("v", len(v))
        if k == 1:
            res_data += v
        else:
            res_data += v * (num // len(v) + 1)
    debug("res_data len", len(res_data))
    return res_data


def get_example(storys):
    Examples = []
    for story in storys:
        Examples.extend(story.rebuild())
    return Examples


def rebuild_example(Examples, emotions_type):
    re_Examples = []
    for item in Examples:
        item.emotions_type = emotions_type
        item.rebuild()
        re_Examples.append(item)
    debug("example", re_Examples[-1].new_character)
    return re_Examples


def read_raw_data(path, args):
    file = read_from_file(path)
    Storys = dict()
    for idx, line in enumerate(file):
        if idx == 0:
            continue
        item = line.strip().split("\t")
        if len(item) == 4:
            id, content, character, emotion = item[0], item[1], item[2], item[3]
        else:
            id, content, character, emotion = item[0], item[1], None, None
        id_split = id.split("_")
        story_id, example_id = str(id_split[0]) + "_" + str(id_split[1]), int(id_split[-1])
        if Storys.get(story_id, None) is None:
            Storys[story_id] = Story(story_id)
        now_storys = Storys[story_id]
        now_storys.add_example(example_id, Example(content, character, emotion, args.emotions_type))
    raw_data = []
    for _, v in Storys.items():
        raw_data.append(v)
    random.shuffle(raw_data)
    return raw_data


def read_file_up(path, args):
    raw_data = read_raw_data(path, args)
    train_story = raw_data[:int(len(raw_data)*0.95)]
    valid_story = raw_data[int(len(raw_data)*0.95):]
    train_data = get_example(train_story)
    valid_data = get_example(valid_story)
    return train_data, valid_data


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
    cls = False
    if args.have_up:
        train_data, valid_data = read_file_up(args.train_path, args)
        # train_data = remake_data(train_data)
    else:
        train_data = read_file(args.train_path, args)
        valid_data = train_data[int(len(train_data)*0.95):]
        train_data = train_data[:int(len(train_data)*0.95)]
        # train_data = remake_data(train_data)
    # _ = remake_data(train_data)
    tokenizer = AutoTokenizer.from_pretrained(args.pre_train_name)
    train_dataset = PromptDataset(train_data, tokenizer, args)
    valid_dataset = PromptDataset(valid_data, tokenizer, args)
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=RerankingCollate(args.fix_length))
    valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=RerankingCollate(args.fix_length))
    model = BERT_prompt(args).cuda()
    logging.info("len: {}".format(len(train_iter)))
    # train(train_iter, valid_iter, model, args)
    train(train_iter, valid_iter, model, args)


def real_main(args):
    train_data, valid_data = read_file_up(args.train_path, args)
    debug("raw data len", len(train_data))
    # kf = KFold(n_splits=10, shuffle=False)
    # kf_ids = 0
    idx = 0
    preffix = args.model_save_path
    args.ernie = True
    # if args.ernie:
    #     tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')
    # else:
    tokenizer = AutoTokenizer.from_pretrained(args.pre_train_name)
    # for _train, _valid in kf.split(raw_data):
        # idx += 1
        # if idx <= 9:
        #     continue
    # train_data = [copy.deepcopy(raw_data[idx]) for idx in _train]
    # valid_data = [copy.deepcopy(raw_data[idx]) for idx in _valid]
    # debug("data type", type(train_data[0]))
    # train_data = get_example(train_data)
    # valid_data = get_example(valid_data)
    # args.model_save_path = preffix + "_KFID_{}".format(kf_ids)
    # kf_ids += 1
    train_dataset = BERTDataset(train_data, tokenizer, args)
    valid_dataset = BERTDataset(valid_data, tokenizer, args)
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=RerankingCollate(args.fix_length))
    valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=RerankingCollate(args.fix_length))
    model = BERT(args).cuda()
    # debug("len train_iter", len(train_iter))
    # debug("train", type(train_iter))
    # logging.info("Start Training KF_IDS:{} Model".format(kf_ids))
    logging.info("train len:{}".format(len(train_iter)))
    train(train_iter, valid_iter, model ,args)


def predict_up_main(args):
    file = read_from_file(args.test_path)
    ans_mp = dict()
    ans_list = []
    predict_list = []
    Storys = dict()
    cnt = 0
    for idx, line in enumerate(file):
        if idx == 0:
            continue
        item = line.strip().split("\t")
        if len(item) == 3:
            id, content, character = item[0], item[1], item[2]
            ans_mp[id] = []
        else:
            id, content, character = item[0], item[1], None
            ans_mp[id] = [0,0,0,0,0,0]
        ans_list.append(id)
        predict_list.append(id)
        id_split = id.split("_")
        story_id, example_id = str(id_split[0]) + "_" + str(id_split[1]), int(id_split[-1])
        if Storys.get(story_id, None) is None:
            Storys[story_id] = Story(story_id)
        now_storys = Storys[story_id]
        now_storys.add_example(example_id, Example(content, character, "0", emotions_type=200, id=cnt))
        cnt += 1
    raw_data = []
    for _, v in Storys.items():
        raw_data.append(v)
    debug("raw_data len", len(raw_data))
    Examples = get_example(raw_data)
    debug("Examples len", len(Examples))
    all = True
    if all:
        model = torch.load(args.model_all).cuda()
        Examples = rebuild_example(Examples, 6)
        tokenizer = BertTokenizer.from_pretrained(args.pre_train_name)
        test_dataset = MyDataset(Examples, tokenizer, args)
        # test_sen = torch.LongTensor([bert_concat_tokenizer(item.new_content, item.new_character, tokenizer, args.fix_length)[0] for item in Examples])
        # test_pad = torch.LongTensor([bert_concat_tokenizer(item.new_content, item.new_character, tokenizer, args.fix_length)[1] for item in Examples])
        # train_id = torch.LongTensor([item.id for item in Examples])
        # test_dataset = torch.utils.data.TensorDataset(test_sen, test_pad, train_id)
        test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
        res = predict(test_iter, model, args)
        for item in res:
            ans_mp[predict_list[int(item[1])]] = item[0]
    else:
        model_list = [args.model_0, args.model_1, args.model_2, args.model_3, args.model_4, args.model_5]
        for idx, model_path in enumerate(model_list):
            logging.info("start predict model_{}".format(idx))
            Examples = rebuild_example(Examples, idx)
            tokenizer = BertTokenizer.from_pretrained(args.pre_train_name)
            test_dataset = MyDataset(Examples, tokenizer, args)
            # test_sen = torch.LongTensor([bert_concat_tokenizer(item.new_content, item.new_character, tokenizer, args.fix_length)[0] for item in Examples])
            # test_pad = torch.LongTensor([bert_concat_tokenizer(item.new_content, item.new_character, tokenizer, args.fix_length)[1] for item in Examples])
            # train_id = torch.LongTensor([item.id for item in Examples])
            # test_dataset = torch.utils.data.TensorDataset(test_sen, test_pad, train_id)
            test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
            model = torch.load(model_path).cuda()
            res = predict(test_iter, model, args)
            for item in res:
                if idx == -1:
                    ans_mp[predict_list[int(item[1])]].append(0)
                else:
                    ans_mp[predict_list[int(item[1])]].append(item[0])
    with open(args.predict_save_path, "w", newline="") as f:
        tsv_w = csv.writer(f, delimiter="\t")
        tsv_w.writerow(["id", "emotion"])
        for id in ans_list:
            tsv_w.writerow([id, list2str(ans_mp[id])])


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
    test_dataset = MyDataset(Examples, tokenizer, args)
    # test_sen = torch.LongTensor([bert_concat_tokenizer(item.new_content, item.new_character, tokenizer, args.fix_length)[0] for item in Examples])
    # test_pad = torch.LongTensor([bert_concat_tokenizer(item.new_content, item.new_character, tokenizer, args.fix_length)[1] for item in Examples])
    # train_id = torch.LongTensor([item.id for item in Examples])
    # test_dataset = torch.utils.data.TensorDataset(test_sen, test_pad, train_id)
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
    logging.info("END...")


def predict_es_main(args):
    file = read_from_file(args.test_path)
    ans_mp = dict()
    ans_list = []
    predict_list = []
    Storys = dict()
    cnt = 0
    for idx, line in enumerate(file):
        if idx == 0:
            continue
        item = line.strip().split("\t")
        if len(item) == 3:
            id, content, character = item[0], item[1], item[2]
            ans_mp[id] = []
        else:
            id, content, character = item[0], item[1], None
            ans_mp[id] = [0,0,0,0,0,0]
        ans_list.append(id)
        predict_list.append(id)
        id_split = id.split("_")
        story_id, example_id = str(id_split[0]) + "_" + str(id_split[1]), int(id_split[-1])
        if Storys.get(story_id, None) is None:
            Storys[story_id] = Story(story_id)
        now_storys = Storys[story_id]
        now_storys.add_example(example_id, Example(content, character, "0", emotions_type=200, id=cnt))
        cnt += 1
    raw_data = []
    for _, v in Storys.items():
        raw_data.append(v)
    debug("raw_data len", len(raw_data))
    Examples = get_example(raw_data)
    model_list = [args.model_all]
    # model_list = [args.model_0, args.model_1, args.model_2, args.model_3, args.model_4,
                #   args.model_5, args.model_6, args.model_7, args.model_8, args.model_9]
    models = [torch.load(model).cuda() for model in model_list]
    Examples = rebuild_example(Examples, 6)
    tokenizer = BertTokenizer.from_pretrained(args.pre_train_name)
    test_dataset = BERTDataset(Examples, tokenizer, args)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=RerankingCollate(args.fix_length, predict=True))
    with torch.no_grad():
        res = predict_kf(test_iter, models, args)
    for item in res:
        # debug("item", item)
        ans_mp[predict_list[int(item[1])]] = item[0]
    with open(args.predict_save_path, "w", newline="") as f:
        tsv_w = csv.writer(f, delimiter="\t")
        tsv_w.writerow(["id", "emotion"])
        for id in ans_list:
            tsv_w.writerow([id, list2str(ans_mp[id])])
    logging.info("END...")


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
    parser.add_argument("--have_up", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--predict_save_path", type=str)
    parser.add_argument("--model_0", type=str)
    parser.add_argument("--model_1", type=str)
    parser.add_argument("--model_2", type=str)
    parser.add_argument("--model_3", type=str)
    parser.add_argument("--model_4", type=str)
    parser.add_argument("--model_5", type=str)
    parser.add_argument("--model_6", type=str)
    parser.add_argument("--model_7", type=str)
    parser.add_argument("--model_8", type=str)
    parser.add_argument("--model_9", type=str)
    parser.add_argument("--model_all", type=str)

    parser.add_argument("--hidden_size", type=int, default=600)
    parser.add_argument("--fix_length_lstm", type=int, default=30)

    args = parser.parse_args()
    logging.info("learning_rate:{}".format(args.learning_rate))
    if args.predict:
        # predict_up_main(args)
        predict_es_main(args)
    else:
        args.model_save_path = "/".join([args.model_save_dir, str(args.emotions_type) + "_" + d2s(datetime.datetime.now(), time=True)])
        # main(args)
        real_main(args)

