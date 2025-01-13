import numpy as np
import torch
import time

from evaluate import cluster_acc, f_score, nmi, ari
from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix


def post_proC(C, K, d, alpha):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5 * (C + C.T)
    start_time = time.time()
    r = d * K + 1  # 对C进行奇异值分解SVD，保留前r=dK+1个最大的奇异值，计算出左奇异矩阵U

    print("svds1 start")
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    print("svds1 end")

    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** alpha)
    L = L / L.max()
    L = 0.5 * (L + L.T)

    print("spectral clustering start")
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    print("spectral clustering end")

    print("fit start")
    spectral.fit(L)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    print("fit end")

    print("fit predict start")
    grp = spectral.fit_predict(L) + 1
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    print("fit end")
    print("fit predict end")

    print("svds2 start")
    uu, ss, vv = svds(L, k=K)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    print("svds2 end")

    return grp, uu  # 返回：样本的聚类结果grp和分解后的子空间uu


def thrC(C, ro):  # 将小于一定阈值ro的元素设为0，返回压缩后的矩阵Cp
    if ro < 1:
        start_time = time.time()
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while stop == False:
                csum = csum + S[t, i]
                if csum > ro * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C
    return Cp


def get_one_hot_Label(Label, num_clusters):
    if Label.min() == 0:
        Label = Label
    else:
        Label = Label - 1

    Label = np.array(Label)
    n_class = num_clusters
    n_sample = Label.shape[0]
    one_hot_Label = np.zeros((n_sample, n_class))
    for i, j in enumerate(Label):
        one_hot_Label[i, j] = 1

    return one_hot_Label


def form_Theta(Q):
    Theta = np.zeros((Q.shape[0], Q.shape[0]))
    for i in range(Q.shape[0]):
        Qq = np.tile(Q[i], [Q.shape[0], 1])
        Theta[i, :] = 1 / 2 * np.sum(np.square(Q - Qq), 1)

    return Theta


def form_structure_matrix(idx, K):
    Q = np.zeros((len(idx), K))
    for i, j in enumerate(idx):
        Q[i, j - 1] = 1

    return Q
