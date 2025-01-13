import argparse
import gc

import torch
from torch import optim
from MvCDSC import Model, IndividualMLPEncoder, CommonMLPEncoder
from utils import *
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import random
import scipy.io as sio
from tqdm import tqdm
import numpy as np
from train import post_proC, thrC, get_one_hot_Label, form_Theta, form_structure_matrix
from evaluate import cluster_acc, nmi, ari, f_score
from train import prepare_graph_data
from sklearn.neighbors import NearestNeighbors
import pickle
import os
from sklearn.preprocessing import normalize
from scipy.fftpack import fft
import time

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='HHAR')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--patience', type=int, default=30)
parser.add_argument('--neg_times', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--generative_flag', type=bool, default=True)
parser.add_argument('--cuda', action='store_true', default=torch.cuda.is_available())
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

device = torch.device("cuda:1")
device2 = torch.device("cuda:0")

if args.dataset == "HHAR":
    args.n_nodes = 10299
    args.feat1 = 561
    args.feat2 = 561
    args.hidden1 = 512
    args.hidden2 = 512
    args.hidden3 = 256
    # args.hidden4 = 256
    args.epoch = 91
    args.n_cluster = 6
    args.pre_epoch = 200
    args.lambda_1 = 0.5
    args.lambda_2 = 10
    args.lambda_3 = 100
    args.lambda_4 = 50
    args.lambda_5 = 0.01
else:
    print("Error!")


def Euler_transform_1D(Data, alpha):
    Data = np.transpose(Data)
    # Data: ndim * Nsamples
    # ndim: data dimension, Nsamples: the number of data
    t_min = np.min(Data, axis=0)     # 每列最小值
    t_max = np.max(Data, axis=0)
    norm_Data = (Data - t_min) / (t_max - t_min)
    E_Data = np.exp(1j*alpha*np.pi*norm_Data) / np.sqrt(2)
    E_Data = np.transpose(E_Data)
    return E_Data


if __name__ == "__main__":
    print('114*514')

    start_time = time.time()

    X = np.load("/home/guzhengtian/xing/data/hhar/hhar_feat.npy")
    y_true = np.load("/home/guzhengtian/xing/data/hhar/hhar_label.npy")

    X2 = Euler_transform_1D(X, 1.1)
    X = torch.from_numpy(X)
    X2 = torch.from_numpy(X2)
    X = X.float()
    X2 = X2.float()

    nbrs = NearestNeighbors(n_neighbors=5 + 1, algorithm='ball_tree').fit(X)
    adj_wave = nbrs.kneighbors_graph(X)  # <class 'scipy.sparse.csr.csr_matrix'>
    A, S, R = prepare_graph_data(adj_wave)

    nbrs2 = NearestNeighbors(n_neighbors=5 + 1, algorithm='ball_tree').fit(X)
    adj_wave2 = nbrs2.kneighbors_graph(X)  # <class 'scipy.sparse.csr.csr_matrix'>
    A2, S2, R2 = prepare_graph_data(adj_wave2)

    X = X.to(device)
    A = A.to(device)
    X2 = X2.to(device)
    A2 = A2.to(device)

    finalModel = Model(args)
    finalModel = finalModel.to(device)
    # check
    for name, param in finalModel.named_parameters():
        print(f"Parameter name: {name}, Parameter device: {param.device}")
    params = [
        {"params": finalModel.enc_layer11.parameters(), "lr": 0.00005},
        {"params": finalModel.enc_layer12.parameters(), "lr": 0.00005},
        {"params": finalModel.weight, "lr": 0.00005},
        {"params": finalModel.dec_layer11.parameters(), "lr": 0.00005},
        {"params": finalModel.dec_layer12.parameters(), "lr": 0.00005},

        {"params": finalModel.enc_layer21.parameters(), "lr": 0.00005},
        {"params": finalModel.enc_layer22.parameters(), "lr": 0.00005},
        {"params": finalModel.weight2, "lr": 0.00005},
        {"params": finalModel.dec_layer21.parameters(), "lr": 0.00005},
        {"params": finalModel.dec_layer22.parameters(), "lr": 0.00005},

        {"params": finalModel.enc_layer31.parameters(), "lr": 0.0001},
        # {"params": finalModel.enc_layer32.parameters(), "lr": 0.0001},
        {"params": finalModel.weight31, "lr": 0.0001},
        {"params": finalModel.weight32, "lr": 0.0001},
        {"params": finalModel.dec_layer31.parameters(), "lr": 0.0001},
        # {"params": finalModel.dec_layer32.parameters(), "lr": 0.0001},
    ]
    finalOptimizer = optim.Adam(params, weight_decay=args.weight_decay)

    pre_model1 = IndividualMLPEncoder(n_fts=args.feat1, args=args)
    optimizer1 = optim.Adam(filter(lambda p: p.requires_grad, pre_model1.parameters()), lr=args.lr,
                           weight_decay=args.weight_decay)
    pre_model1 = pre_model1.to(device)
    # check
    print("-------------------------------------------------------------")
    for name, param in pre_model1.named_parameters():
        print(f"Parameter name: {name}, Parameter device: {param.device}")

    for epoch in range(250):
        pre_model1.train()
        optimizer1.zero_grad()
        H1, loss1, ft_loss1, st_loss1, SE_loss1, C_Regular1 = pre_model1(X, A, S, R, args)
        L1 = loss1
        L1.backward()
        optimizer1.step()
        if epoch % 100 == 0:
            print("-------------------------------------------------------------")
            print("pre_epoch: %d" % epoch, "Pre_Loss: %.2f" % loss1, "Pre_ft_loss: %.2f" % ft_loss1,
                  "Pre_st_loss: %.2f" % st_loss1,
                  "Pre_SE_loss: %.2f" % SE_loss1,
                  "Pre_C_Regular: %.2f" % C_Regular1)
    H1, loss1, ft_loss1, st_loss1, SE_loss1, C_Regular1 = pre_model1(X, A, S, R, args)

    finalModel.enc_layer11.load_state_dict(pre_model1.encoder1.state_dict())
    finalModel.enc_layer12.load_state_dict(pre_model1.encoder2.state_dict())
    finalModel.weight.data.copy_(pre_model1.weight.data)
    finalModel.dec_layer11.load_state_dict(pre_model1.decoder1.state_dict())
    finalModel.dec_layer12.load_state_dict(pre_model1.decoder2.state_dict())

    del loss1, ft_loss1, st_loss1, SE_loss1, C_Regular1, pre_model1
    torch.cuda.empty_cache()
    gc.collect()

    pre_model2 = IndividualMLPEncoder(n_fts=args.feat2, args=args)
    optimizer2 = optim.Adam(filter(lambda p: p.requires_grad, pre_model2.parameters()), lr=args.lr,
                            weight_decay=args.weight_decay)
    pre_model2 = pre_model2.to(device)
    # check
    print("-------------------------------------------------------------")
    for name, param in pre_model2.named_parameters():
        print(f"Parameter name: {name}, Parameter device: {param.device}")

    for epoch in range(250):
        pre_model2.train()
        optimizer2.zero_grad()
        H2, loss2, ft_loss2, st_loss2, SE_loss2, C_Regular2 = pre_model2(X2, A2, S2, R2, args)
        L2 = loss2
        L2.backward()
        optimizer2.step()
        if epoch % 100 == 0:
            print("-------------------------------------------------------------")
            print("pre_epoch: %d" % epoch, "Pre_Loss: %.2f" % loss2, "Pre_ft_loss: %.2f" % ft_loss2,
                  "Pre_st_loss: %.2f" % st_loss2,
                  "Pre_SE_loss: %.2f" % SE_loss2,
                  "Pre_C_Regular: %.2f" % C_Regular2)
    H2, loss2, ft_loss2, st_loss2, SE_loss2, C_Regular2 = pre_model2(X2, A2, S2, R2, args)

    finalModel.enc_layer21.load_state_dict(pre_model2.encoder1.state_dict())
    finalModel.enc_layer22.load_state_dict(pre_model2.encoder2.state_dict())
    finalModel.weight2.data.copy_(pre_model2.weight.data)
    finalModel.dec_layer21.load_state_dict(pre_model2.decoder1.state_dict())
    finalModel.dec_layer22.load_state_dict(pre_model2.decoder2.state_dict())

    del loss2, ft_loss2, st_loss2, SE_loss2, C_Regular2, pre_model2
    torch.cuda.empty_cache()
    gc.collect()

    pre_model3 = CommonMLPEncoder(args=args)
    optimizer3 = optim.Adam(filter(lambda p: p.requires_grad, pre_model3.parameters()), lr=args.lr,
                            weight_decay=args.weight_decay)
    pre_model3 = pre_model3.to(device)
    # check
    print("-------------------------------------------------------------")
    for name, param in pre_model3.named_parameters():
        print(f"Parameter name: {name}, Parameter device: {param.device}")

    for epoch in range(200):
        pre_model3.train()
        optimizer3.zero_grad()
        coef1, coef2, loss3, ft_loss3, st_loss3, SE_loss3, C_Regular3 = pre_model3(H1, A, S, R, H2, A2, S2, R2, args)
        L3 = loss3
        L3.backward(retain_graph=True)
        optimizer3.step()
        if epoch % 100 == 0:
            print("-------------------------------------------------------------")
            print("pre_epoch: %d" % epoch, "Pre_Loss: %.2f" % loss3, "Pre_ft_loss: %.2f" % ft_loss3,
                  "Pre_st_loss: %.2f" % st_loss3,
                  "Pre_SE_loss: %.2f" % SE_loss3,
                  "Pre_C_Regular: %.2f" % C_Regular3)
    coef1, coef2, loss3, ft_loss3, st_loss3, SE_loss3, C_Regular3 = pre_model3(H1, A, S, R, H2, A2, S2, R2, args)

    del loss3, ft_loss3, st_loss3, SE_loss3, C_Regular3
    torch.cuda.empty_cache()
    gc.collect()

    finalModel.enc_layer31.load_state_dict(pre_model3.encoder1.state_dict())
    # finalModel.enc_layer32.load_state_dict(pre_model3.encoder2.state_dict())
    finalModel.weight31.data.copy_(pre_model3.weight1.data)
    finalModel.weight32.data.copy_(pre_model3.weight2.data)
    finalModel.dec_layer31.load_state_dict(pre_model3.decoder1.state_dict())
    # finalModel.dec_layer32.load_state_dict(pre_model3.decoder2.state_dict())

    del pre_model3
    torch.cuda.empty_cache()
    gc.collect()

    alpha = max(0.4 - (6 - 1) / 10 * 0.1, 0.1)
    coef_ans = 0.7 * coef1 + 0.3 * coef2
    commonZ = thrC(coef_ans.cpu().detach().numpy(), alpha)
    y_x, _ = post_proC(commonZ, args.n_cluster, 10, 3.5)
    s2_label_subjs = np.array(y_x)
    s2_label_subjs = s2_label_subjs - s2_label_subjs.min() + 1
    s2_label_subjs = np.squeeze(s2_label_subjs)
    one_hot_Label = get_one_hot_Label(s2_label_subjs, args.n_cluster)
    s2_Q = form_structure_matrix(s2_label_subjs, args.n_cluster)
    s2_Theta = form_Theta(s2_Q)
    s2_Theta = torch.from_numpy(s2_Theta)
    Y = y_x
    Y = Y.reshape(args.n_nodes, 1)

    print("-------------------------------------------------------------")
    print("Initial Clustering Results: ")
    print("acc: {:.8f}\t\tnmi: {:.8f}\t\tf_score: {:.8f}\t\tari: {:.8f}".
          format(cluster_acc(y_true, y_x - 1), nmi(y_true, y_x - 1), f_score(y_true, y_x - 1),
                 ari(y_true, y_x - 1)))
    print("-------------------------------------------------------------")

    del coef1, coef2
    torch.cuda.empty_cache()
    gc.collect()

    for epoch in range(args.epoch):
        finalModel.train()
        finalOptimizer.zero_grad()
        loss, ft_loss, st_loss, SE_loss, C_Regular, consistent_loss, cl_loss, Cq_loss, coef3 = finalModel(X, A, S, R, X2, A2, S2, R2, Y-1, s2_Theta)
        loss.backward()
        finalOptimizer.step()

        if epoch % 10 == 0:
            coef_ans = coef3
            commonZ = thrC(coef_ans.cpu().detach().numpy(), alpha)
            y_x, _ = post_proC(commonZ, args.n_cluster, 10, 3.5)
            print("Epoch--{}:\t\tloss: {:.8f}\t\tacc: {:.8f}\t\tnmi: {:.8f}\t\tf1: {:.8f}\t\tari: {:.8f}".
                  format(epoch, loss, cluster_acc(y_true, y_x - 1), nmi(y_true, y_x - 1),
                         f_score(y_true, y_x - 1), ari(y_true, y_x - 1)))
            print("Epoch--{}:\t\tft_loss: {:.8f}\t\tst_loss: {:.8f}\t\tSE_loss: {:.8f}\t\tC_Regular: {:.8f}"
                  "\t\tcl_loss: {:.8f}\t\tCq_loss: {:.8f}\t\tconsistent_loss: {:.8f}".
                  format(epoch, ft_loss, st_loss, SE_loss, C_Regular, cl_loss, Cq_loss, consistent_loss))

        if epoch % 20 == 0:
            # coef_ans = coef3
            # commonZ = thrC(coef_ans.cpu().detach().numpy(), alpha)
            # y_x, _ = post_proC(commonZ, args.n_cluster, 10, 3.5)
            Y = y_x
            Y = Y.reshape(args.n_nodes, 1)
            s2_label_subjs = np.array(Y)
            s2_label_subjs = s2_label_subjs - s2_label_subjs.min() + 1
            s2_label_subjs = np.squeeze(s2_label_subjs)
            one_hot_Label = get_one_hot_Label(s2_label_subjs, args.n_cluster)
            s2_Q = form_structure_matrix(s2_label_subjs, args.n_cluster)
            s2_Theta = form_Theta(s2_Q)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time elapsed: {elapsed_time} seconds")








