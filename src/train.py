import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
cnt = 0

def save(model, path):
    global cnt
    path += "_{}.pkl".format(cnt)
    cnt += 1
    if os.path.exists(path):
        os.remove(path)
        logging.info("model remove success!!!")
    logging.info("Save model")
    torch.save(model, path)


def real_res(logit, T=True):
    x = logit.tolist()
    res = []
    for item in x:
        if item < 0.8:
            res.append(0)
        elif item < 1.8:
            res.append(1)
        elif item < 2.8:
            res.append(2)
        else:
            res.append(3)
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
        logit = logit.cpu().reshape(id.shape)
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
    for sen, pad, label in valid_iter:
        # debug("sen", sen)
        logit = model(sen.cuda(), pad.cuda())
        logit = logit.cpu().reshape(label.shape)
        logit = real_res(logit * 3)
        # logging.debug(logit.cpu())
        # logging.debug(torch.max(logit.cpu(), 1)[1].view(label.size()))
        debug("logit", logit)
        debug("label", label * 3)
        mse = LOSS(logit, label * 3)
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
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    need_eval = 1
    rmse_max = 0
    LOSS = nn.MSELoss()
    for step in range(args.epoch):
        loss_sum = 0
        loss_s_sum = 0
        model.train()
        for sen, pad, label in train_iter:
            logit = model(sen.cuda(), pad.cuda())
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
                    if rmse_n > rmse_max:
                        rmse_max = rmse_n
                        save(model, args.model_save_path)
                model.train()
            need_eval += 1
        # logging.info("loss:{:.4f}".format(loss_sum / len(train_iter)))
        with torch.no_grad():
            rmse_n = eval(valid_iter, model, args)
            if rmse_n > rmse_max:
                rmse_max = rmse_n
                save(model, args.model_save_path)
    logging.info("Finished Training!!!")
