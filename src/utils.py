import pandas as pd
import logging


def bert_concat_tokenizer_list(sen1, sen2, prompt, tokenizer, fix_length):
    if len(sen1) + len(sen2) + 30 > fix_length:
        sen1 = sen1[-(fix_length-30-len(sen2)):]
    sen1_ids = tokenizer.encode(sen1, add_special_tokens=False)
    sen2_ids = tokenizer.encode(sen2, add_special_tokens=False)
    ids = []
    mask = []
    token_type_ids = []
    for sen in prompt:
        sen3_ids = tokenizer.encode(sen, add_special_tokens=False) 
        ids.append([101] + sen1_ids + [102] + sen2_ids + sen3_ids + [102])
        mask.append([1] * len(ids[-1]))
        token_type_ids.append([0] * (len(sen1_ids) + 2) + [1] * (len(sen2_ids) + len(sen3_ids) + 1))
    return ids, mask, token_type_ids


def bert_concat_tokenizer_new(sen1, sen2, tokenizer, fix_length):
    if len(sen1) > 500:
        sen1 = sen1[-500:]
    if len(sen2) > 500:
        sen2 = sen2[-500:]    
    sen1_ids = tokenizer.encode(sen1, add_special_tokens=False)
    sen2_ids = tokenizer.encode(sen2, add_special_tokens=False)
    if len(sen1_ids) + len(sen2_ids) > fix_length -3:
        need = fix_length -3 - len(sen1_ids)
        sen2_ids = sen2_ids[-need:]
    ids = [101] + sen1_ids + [102] + sen2_ids + [102]
    token_type_ids = [0] * (len(sen1_ids) + 2) + [1] * (len(sen2_ids) + 1)
    mask = [1] * len(ids)
    return ids, mask, token_type_ids


def bert_concat_tokenizer(sen1, sen2, tokenizer, fix_length):
    if len(sen1) > 500:
        sen1 = sen1[-500:]
    sen1_ids = tokenizer.encode(sen1, add_special_tokens=False)
    sen2_ids = tokenizer.encode(sen2, add_special_tokens=False)
    if len(sen1_ids) + len(sen2_ids) > fix_length -3:
        need = fix_length -3 - len(sen2_ids)
        sen1_ids = sen1_ids[-need:]
    sen_ids = [101] + sen1_ids + [102] + sen2_ids + [102]
    padding = [1] * len(sen_ids) 
    sen_ids = sen_ids + [0] * (fix_length - len(sen_ids))
    padding += [0] * (len(sen_ids) - len(padding))
    return sen_ids, padding


def bert_concat_id_tokenize(sen1, sen2, tokenizer, fix_length):
    if len(sen1) > 500:
        sen1 = sen1[-500:]
    sen1_ids = tokenizer.encode(sen1, add_special_tokens=False)
    sen2_ids = tokenizer.encode(sen2, add_special_tokens=False)
    if len(sen1_ids) + len(sen2_ids) > fix_length -3:
        need = fix_length -3 - len(sen2_ids)
        sen1_ids = sen1_ids[-need:]
    sen_ids = [101] + sen1_ids + [102] + sen2_ids + [102]
    sentence_id = [0] * (len(sen1_ids) + 2) + [1] * (len(sen2_ids) + 1)
    padding = [1] * len(sen_ids)
    start = len(sen1_ids) + 2
    end = len(sen_ids)
    sentence_id += [0] * (fix_length - len(sen_ids))
    sen_ids = sen_ids + [1] * (fix_length - len(sen_ids))
    padding += [0] * (len(sen_ids) - len(padding))
    return sen_ids, padding, sentence_id, start, end


def bert_tokenizer(sen, tokenizer, fix_length):
    sen_ids = tokenizer.encode(sen, add_special_tokens=False)
    while len(sen_ids) > fix_length - 2:
        sen_ids.pop()
    sen_ids = [101] + sen_ids + [102]
    padding = [1] * len(sen_ids)
    sen_ids = sen_ids + [0] * (fix_length - len(sen_ids))
    padding = padding + [0] * (fix_length - len(padding))
    return sen_ids, padding


def read_from_tsv(path):
    tsv = pd.read_csv(path, sep="\t")
    return tsv


def read_from_file(path):
    with open(path, "r", encoding="utf-8") as f:
        files = f.readlines()
    return files


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


def shuffle_list(list):
    pass