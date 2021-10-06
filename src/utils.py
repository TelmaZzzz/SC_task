import pandas as pd
import logging


def bert_concat_tokenizer(sen1, sen2, tokenizer, fix_length):
    sen1_ids = tokenizer.encode(sen1, add_special_tokens=False)
    sen2_ids = tokenizer.encode(sen2, add_special_tokens=False)
    while len(sen1_ids) + len(sen2_ids) > fix_length - 3:
        if len(sen1_ids) > len(sen2_ids):
            sen1_ids.pop()
        else:
            sen2_ids.pop()
    sen_ids = [101] + sen1_ids + [102] + sen2_ids + [102]
    padding = [1] * len(sen_ids) 
    sen_ids = sen_ids + [0] * (fix_length - len(sen_ids))
    padding += [0] * (len(sen_ids) - len(padding))
    return sen_ids, padding


def bert_tokenizer(sen, tokenizer, fix_length):
    sen_ids = tokenizer.encode(sen, add_special_tokens=False)
    while len(sen_ids) > fix_length - 2:
        sen_ids.pop()
    sen_ids = [101] + sen_ids + [102]
    sen_ids = sen_ids + [0] * (fix_length - len(sen_ids))
    padding = [1] * len(sen_ids)
    return sen_ids, padding


def read_from_tsv(path):
    tsv = pd.read_csv(path, sep="\t")
    return tsv


def d2s(dt, time=False):
    if time is False:
        return dt.strftime("%Y_%m_%d")
    else:
        return dt.strftime("%Y_%m_%d_%H_%M")


def debug(name, item):
    logging.debug("{}: {}".format(name, item))


def list2str(list):
    res = ""
    for item in list:
        res+=str(item)
        res+=","
    return res[:-1]