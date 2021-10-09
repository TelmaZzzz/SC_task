import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


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


def predict(test_iter, model, args):
    model.eval()
    mse_sum = 0
    total = 0
    LOSS = nn.MSELoss()
    ans = []
    ids = []
    for sen, pad, id in test_iter:
        # debug("sen", sen)
        logit = model(sen.cuda(), pad.cuda())
        logit = logit.cpu()
        # debug("logit", logit.cpu())
        logit = real_res(logit * 3, T=False)
        ids += id.tolist()
        ans += logit
    return zip(ans,ids)
        # logging.debug(logit.cpu())
        # logging.debug(torch.max(logit.cpu(), 1)[1].view(label.size()))
        


def eval(valid_iter, model, args):
    logging.info("Start Eval...")
    model.eval()
    mse_sum = 0
    total = 0
    LOSS = nn.MSELoss()
    cls = False
    for sen, pad, label in valid_iter:
        # debug("sen", sen)
        logit = model(sen.cuda(), pad.cuda())
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
    return score


def train(train_iter, valid_iter, model, args):
    logging.info("Start training...")
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if 'bert' not in n]}, 
        {'params': [p for n, p in model.named_parameters() if 'bert' in n], 'lr': args.learning_rate * 0.01}
    ]

    optimizer = torch.optim.Adam(optimizer_grouped_parameters, args.learning_rate, weight_decay=0.0001)
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
        for sen, pad, label in train_iter:
            logit = model(sen.cuda(), pad.cuda())
            debug("logit size", logit.cpu().size())
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
    logging.info("Finished Training!!!")
