import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from transformers import get_linear_schedule_with_warmup, AdamW


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def judge(score_save, score):
    if len(score_save) < 5:
        return -1
    minn = score_save[0]
    pos = 0
    for idx, item in enumerate(score_save):
        if minn > item:
           pos = idx
           minn = item
    if minn > score:
        return -2
    return pos 


def save(model, path, score):
    path += "_{}.pkl".format("{:.4f}".format(score))
    # if os.path.exists(path):
    #     os.remove(path)
    #     logging.info("model remove success!!!")
    logging.info("Save model")
    torch.save(model, path)


def judge_type(x):
    if x < 0.5:
        return 0
    elif x < 1.5:
        return 1
    elif x < 2.5:
        return 2
    else:
        return 3


def real_res(logit, T=True):
    x = logit.tolist()
    res = []
    for item in x:
        if type(item) is list:
            rres = []
            for it in item:
                rres.append(judge_type(it))
            res.append(rres)
            continue
        res.append(judge_type(it))
    if T:
        return torch.Tensor(res)
    else:
        return res


def predict_kf(test_iter, models, args):
    for model in models:
        model.eval()
    id_list = []
    ans = []
    for item in test_iter:
        ids = item["ids"]
        mask = item["mask"]
        token_type_ids = item["token_type_ids"]
        id = item["label"].view(-1)
        sum_logit = None
        for model in models:
            logit = model(ids.cuda(), mask.cuda(), token_type_ids.cuda())
            if sum_logit is None:
                sum_logit = logit.cpu()*3
            else:
                sum_logit = sum_logit + logit.cpu()*3
        sum_logit = sum_logit / len(models)
        logit = real_res(sum_logit, T=False)
        # debug("logit", logit)
        # debug("id", id.tolist())
        id_list += id.tolist()
        ans += logit
    return zip(ans, id_list)


def predict(test_iter, model, args):
    model.eval()
    mse_sum = 0
    total = 0
    LOSS = nn.MSELoss()
    ans = []
    ids = []
    for item in test_iter:
        # debug("sen", sen)
        new_version = True
        if new_version:
            sen = item[0]
            sen_id = item[1]
            pad = item[2]
            start_end = item[3]
            id = item[4].view(-1)
            character = item[5]
            logit = model(sen.cuda(), sen_id.cuda(), pad.cuda(), start_end, character)
        else:
            sen = item[0]
            pad = item[1]
            id = item[2]
            logit = model(sen.cuda(), pad.cuda())
        logit = logit.cpu()
        # debug("logit", logit.cpu())
        logit = real_res(logit * 3, T=False)
        ids += id.tolist()
        debug("logit", logit)
        debug("id", id.tolist())
        ans += logit
    return zip(ans,ids)
        # logging.debug(logit.cpu())
        # logging.debug(torch.max(logit.cpu(), 1)[1].view(label.size()))
        


def eval(valid_iter, model, args, ema=None):
    logging.info("Start Eval...")
    # ema.apply_shadow()
    model.eval()
    mse_sum = 0
    total = 0
    LOSS = nn.MSELoss()
    cls = False
    for item in valid_iter:
        # debug("sen", sen)
        new_version = False
        if new_version:
            sen = item[0]
            sen_id = item[1]
            pad = item[2]
            start_end = item[3]
            label = item[4]
            character = item[5]
            logit = model(sen.cuda(), sen_id.cuda(), pad.cuda(), start_end, character)
        else:
            ids = item["ids"]
            mask = item["mask"]
            label = item["label"]
            token_type_ids = item["token_type_ids"]
            logit = model(ids.cuda(), mask.cuda(), token_type_ids.cuda())
        if cls:
            logit = torch.max(logit.cpu(), 1)[1].view(label.size())
            mse = LOSS(1.0*logit, 1.0*label)
        else:
            logit = logit.cpu().reshape(label.shape)
            logit = real_res(logit * 3).view(-1)
            label = (label * 3).view(-1)
            # logging.debug(logit.cpu())
            # logging.debug(torch.max(logit.cpu(), 1)[1].view(label.size()))
            debug("logit", logit)
            debug("label", label * 3)
            mse = LOSS(logit, label)
            debug("mse", mse)
        mse_sum += mse * label.size(0)
        total += label.size(0)
    rmse = torch.sqrt(mse_sum / total)
    score = 1 / (1 + rmse.item())
    logging.info("rmse: {:.4f}".format(rmse))
    logging.info("score: {:.4f}".format(score))
    logging.info("Finished eval!!!")
    # ema.restore()
    return score


def train(train_iter, valid_iter, model, args):
    logging.info("Start training...")
    need_fen = True
    need_ema = True
    # ema = EMA(model, 0.999)
    # ema.register()
    if need_fen:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if 'bert' not in n]}, 
            {'params': [p for n, p in model.named_parameters() if 'bert' in n], 'lr': args.learning_rate * 0.01}
        ]
    else:
        optimizer_grouped_parameters = model.parameters()
    optimizer = AdamW(optimizer_grouped_parameters, args.learning_rate, weight_decay=0.0001, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_iter) * 2, num_training_steps=len(train_iter) * (args.epoch-2))
    need_eval = 1
    rmse_max = 0
    cls = False
    LOSS = nn.MSELoss()
    score_save = []
    for step in range(args.epoch):
        logging.info("Start train epoch :{}".format(step))
        loss_sum = 0
        loss_s_sum = 0
        model.train()
        for item in train_iter:
            new_version = False
            if new_version:
                sen = item[0]
                sen_id = item[1]
                pad = item[2]
                start_end = item[3]
                label = item[4]
                character = item[5]
                logit = model(sen.cuda(), sen_id.cuda(), pad.cuda(), start_end, character)
            else:
                ids = item["ids"]
                mask = item["mask"]
                label = item["label"]
                token_type_ids = item["token_type_ids"]
                debug("ids size", ids.size())
                debug("mask size", mask.size())
                debug("token_type_ids size", token_type_ids.size())
                logit = model(ids.cuda(), mask.cuda(), token_type_ids.cuda())
            debug("logit size", logit.size())
            debug("label size", label.size())
            if cls:
                try:
                    loss = F.cross_entropy(logit.cpu(), label.squeeze(-1))
                except:
                    optimizer.zero_grad()
                    optimizer.step()
                    continue
            else:
                loss = LOSS(logit.cpu().reshape(label.shape), label)
            # loss_sum += loss.item()
            # debug("train_logit", logit.cpu())
            # debug("train_label", label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            # ema.update()
            # debug("loss", loss)
            if need_eval % args.eval_step == 0:
                # logging.info("loss:{:.4f}".format(loss_s_sum / args.eval_step))
                loss_s_sum = 0
                with torch.no_grad():
                    rmse_n = eval(valid_iter, model, args)
                    pos = judge(score_save, rmse_n)
                    if pos == -1:
                        score_save.append(rmse_n)
                        save(model, args.model_save_path, rmse_n)
                    elif pos != -2:
                        path = args.model_save_path + "_{}.pkl".format("{:.4f}".format(score_save[pos]))
                        if os.path.exists(path):
                            os.remove(path)
                            logging.info("model remove success!!!")
                        score_save = score_save[:pos] + score_save[pos+1:]
                        score_save.append(rmse_n)
                        save(model, args.model_save_path, rmse_n)
                model.train()
            need_eval += 1
        # logging.info("loss:{:.4f}".format(loss_sum / len(train_iter)))
        # logging.info("params: {}".format(optimizer.state_dict()))
        with torch.no_grad():
            rmse_n = eval(valid_iter, model, args)
            pos = judge(score_save, rmse_n)
            if pos == -1:
                score_save.append(rmse_n)
                # save(model, args.model_save_path, rmse_n)
            elif pos != -2:
                path = args.model_save_path + "_{}.pkl".format("{:.4f}".format(score_save[pos]))
                if os.path.exists(path):
                    os.remove(path)
                    logging.info("model remove success!!!")
                score_save = score_save[:pos] + score_save[pos+1:]
                score_save.append(rmse_n)
                # save(model, args.model_save_path, rmse_n)
    logging.info("Finished Training!!!")
