import tensorflow as tf
import numpy as np
from model.MvCDSC import MvCDSC
from model.SinGATE import SINGATE
from model.DouGATE import DOUGATE
from utils.evaluate import cluster_acc, f_score, nmi, ari, err_rate
from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.preprocessing import normalize


def post_proC(C, K, d, alpha):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5 * (C + C.T)
    r = d * K + 1  # 对C进行奇异值分解SVD，保留前r=dK+1个最大的奇异值，计算出左奇异矩阵U
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
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
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    uu, ss, vv = svds(L, k=K)
    return grp, uu  # 返回：样本的聚类结果grp和分解后的子空间uu


def thrC(C, ro):  # 将小于一定阈值ro的元素设为0，返回压缩后的矩阵Cp
    if ro < 1:
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


class Trainer():
    def __init__(self, args):
        self.args = args
        self.build_placeholders()
        self.ae1 = SINGATE(self.args.lambda_1, self.args.lambda_2, self.args.hidden_dims1, self.args.n_sample,
                           self.args.cluster, self.args.alpha, 1)
        self.pre_loss1, self.ft_loss1, self.st_loss1, self.SE_loss1, self.C_Regular1, self.Z1 = self.ae1(self.A, self.X, self.R, self.S)
        self.ae2 = SINGATE(self.args.lambda_1, self.args.lambda_2, self.args.hidden_dims2, self.args.n_sample,
                           self.args.cluster, self.args.alpha, 2)
        self.pre_loss2, self.ft_loss2, self.st_loss2, self.SE_loss2, self.C_Regular2, self.Z2 = self.ae2(self.A2, self.X2, self.R2, self.S2)
        self.ae3 = DOUGATE(self.args.lambda_1, self.args.lambda_2, self.args.hidden_dims3, self.args.n_sample,
                           self.args.cluster, self.args.alpha)
        self.pre_loss3, self.ft_loss3, self.st_loss3, self.SE_loss3, self.C_Regular3 = self.ae3(self.A, self.X, self.R,
                                                                                                self.S, self.A2,
                                                                                                self.X2, self.R2,
                                                                                                self.S2)
        self.model = MvCDSC(self.args)

        all_vars = tf.global_variables()
        for var in all_vars:
            print(var)

        self.vars_1 = [var for var in all_vars if "sin1" in var.name]
        self.vars_2 = [var for var in all_vars if "sin2" in var.name]
        self.vars_3 = [var for var in all_vars if "dou" in var.name]
        self.vars_4 = [var for var in all_vars if "DP1" in var.name]
        self.vars_5 = [var for var in all_vars if "DP2" in var.name]
        self.vars_6 = [var for var in all_vars if "DP3" in var.name]
        self.vars_7 = [var for var in all_vars if ("DP1" in var.name) or ("DP2" in var.name)]

        self.loss, self.ft_loss, self.st_loss, self.SE_loss, self.C_Regular, self.consistent_loss, self.cl_loss, self.Cq_loss, \
            self.H, self.C, self.H2, self.C2, self.z, self.z2, self.coef3 = \
            self.model(self.A, self.X,
                       self.R, self.S, self.p, self.y_pred,
                       self.A2, self.X2, self.R2, self.S2, self.Theta)
        self.pre_optimize1(self.pre_loss1, self.vars_1)
        self.pre_optimize2(self.pre_loss2, self.vars_2)
        self.pre_optimize3(self.pre_loss3, self.vars_3)

        self.optimize(self.loss, self.vars_7, self.vars_6)
        self.build_session()

    def build_placeholders(self):  # placeholder为占位符，类似于函数参数，运行时必须传入值，代表训练或测试数据
        self.A = tf.sparse_placeholder(dtype=tf.float32)
        self.X = tf.placeholder(dtype=tf.float32)
        self.S = tf.placeholder(dtype=tf.int64)
        self.R = tf.placeholder(dtype=tf.int64)
        self.A2 = tf.sparse_placeholder(dtype=tf.float32)
        self.X2 = tf.placeholder(dtype=tf.float32)
        self.S2 = tf.placeholder(dtype=tf.int64)
        self.R2 = tf.placeholder(dtype=tf.int64)
        self.p = tf.placeholder(dtype=tf.float32, shape=(None, self.args.cluster))
        self.y_pred = tf.placeholder(tf.float32, [self.args.n_sample, 1])
        self.Theta = tf.placeholder(tf.float32, [self.args.n_sample, self.args.n_sample])

    # tensorflow中所有的计算步骤（包括数据定义等等）都要在一个graph中事先定义好，然后通过session来执行这个graph（全部或部分），得到结果
    # 通常需要通过定义placeholder、Variable和OP等构成一张完整的计算图Graph，接下来通过新建Session实例启动模型运行
    # Session实例会分布式执行Graph，输入数据，根据优化算法更新Variable，然后返回执行结果即Tensor实例
    def build_session(self):
        config = tf.ConfigProto()  # 定义Session运行配置，例如指定设备是否自动分配等
        config.gpu_options.allow_growth = True
        # config.intra_op_parallelism_threads = 0
        # config.inter_op_parallelism_threads = 0
        self.session = tf.Session(config=config)
        self.session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        # tf.global_variables_initializer()是初始化模型的参数
        # tf.local_variables_initializer()返回一个初始化所有局部变量的操作（OP）

    def pre_optimize1(self, pre_loss1, vars_1):
        pre_optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
        pre_gradients, pre_variables = zip(*pre_optimizer.compute_gradients(pre_loss1, var_list=vars_1))
        pre_gradients, _ = tf.clip_by_global_norm(pre_gradients, self.args.gradient_clipping)
        self.pre_train_op1 = pre_optimizer.apply_gradients(zip(pre_gradients, pre_variables))

    def pre_optimize2(self, pre_loss2, vars_2):
        pre_optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
        pre_gradients, pre_variables = zip(*pre_optimizer.compute_gradients(pre_loss2, var_list=vars_2))
        pre_gradients, _ = tf.clip_by_global_norm(pre_gradients, self.args.gradient_clipping)
        self.pre_train_op2 = pre_optimizer.apply_gradients(zip(pre_gradients, pre_variables))

    def pre_optimize3(self, pre_loss3, vars_3):
        pre_optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
        pre_gradients, pre_variables = zip(*pre_optimizer.compute_gradients(pre_loss3, var_list=vars_3))
        pre_gradients, _ = tf.clip_by_global_norm(pre_gradients, self.args.gradient_clipping)
        self.pre_train_op3 = pre_optimizer.apply_gradients(zip(pre_gradients, pre_variables))

    def optimize(self, loss, vars_7, vars_6):
        optimizer1 = tf.train.AdamOptimizer(learning_rate=self.args.pre_lr)
        gradients1, variables1 = zip(*optimizer1.compute_gradients(loss, var_list=vars_7))
        gradients1, _ = tf.clip_by_global_norm(gradients1, self.args.gradient_clipping)
        train_op1 = optimizer1.apply_gradients(zip(gradients1, variables1))

        optimizer2 = tf.train.AdamOptimizer(learning_rate=1e-4)
        gradients2, variables2 = zip(*optimizer2.compute_gradients(loss, var_list=vars_6))
        gradients2, _ = tf.clip_by_global_norm(gradients2, self.args.gradient_clipping)
        train_op2 = optimizer2.apply_gradients(zip(gradients2, variables2))
        self.train_op = tf.group(train_op1, train_op2)

    def __call__(self, A, X, S, R, A2, X2, S2, R2, y_true, num_clusters):
        for epoch in range(500):
            pre_loss1, _, pre_st_loss1, pre_ft_loss1, pre_C_Regular1, pre_SE_loss1, Z1 = self.session.run(
                [self.pre_loss1, self.pre_train_op1, self.st_loss1, self.ft_loss1, self.C_Regular1, self.SE_loss1,
                 self.Z1], feed_dict={self.A: A, self.X: X, self.R: R, self.S: S})
            if epoch % 100 == 0:
                print("-------------------------------------------------------------")
                print("pre_epoch: %d" % epoch, "Pre_Loss: %.2f" % pre_loss1, "Pre_ft_loss: %.2f" % pre_ft_loss1,
                      "Pre_st_loss: %.2f" % pre_st_loss1, "Pre_SE_loss: %.2f" % pre_SE_loss1,
                      "Pre_C_Regular: %.2f" % pre_C_Regular1)

        for epoch in range(500):
            pre_loss2, _, pre_st_loss2, pre_ft_loss2, pre_C_Regular2, pre_SE_loss2, Z2 = self.session.run(
                [self.pre_loss2, self.pre_train_op2, self.st_loss2, self.ft_loss2, self.C_Regular2, self.SE_loss2,
                 self.Z2], feed_dict={self.A2: A2, self.X2: X2, self.R2: R2, self.S2: S2})
            if epoch % 100 == 0:
                print("-------------------------------------------------------------")
                print("pre_epoch: %d" % epoch, "Pre_Loss: %.2f" % pre_loss2, "Pre_ft_loss: %.2f" % pre_ft_loss2,
                      "Pre_st_loss: %.2f" % pre_st_loss2, "Pre_SE_loss: %.2f" % pre_SE_loss2,
                      "Pre_C_Regular: %.2f" % pre_C_Regular2)

        for epoch in range(300):
            pre_loss3, _, pre_st_loss3, pre_ft_loss3, pre_C_Regular3, pre_SE_loss3 = self.session.run(
                [self.pre_loss3, self.pre_train_op3, self.st_loss3, self.ft_loss3, self.C_Regular3, self.SE_loss3],
                feed_dict={self.A: A, self.X: Z1, self.R: R, self.S: S,
                           self.A2: A2, self.X2: Z2, self.R2: R2, self.S2: S2})
            if epoch % 100 == 0:
                print("-------------------------------------------------------------")
                print("pre_epoch: %d" % epoch, "Pre_Loss: %.2f" % pre_loss3, "Pre_ft_loss: %.2f" % pre_ft_loss3,
                      "Pre_st_loss: %.2f" % pre_st_loss3, "Pre_SE_loss: %.2f" % pre_SE_loss3,
                      "Pre_C_Regular: %.2f" % pre_C_Regular3)

        # assign the pretrained model's weights to the trained model
        copy_ops = []
        for vars_1, vars_4 in zip(self.vars_1, self.vars_4):
            copy_op = tf.assign(vars_4, vars_1)
            copy_ops.append(copy_op)
        self.session.run(copy_ops)

        copy_ops = []
        for vars_2, vars_5 in zip(self.vars_2, self.vars_5):
            copy_op = tf.assign(vars_5, vars_2)
            copy_ops.append(copy_op)
        self.session.run(copy_ops)

        copy_ops = []
        for vars_3, vars_6 in zip(self.vars_3, self.vars_6):
            copy_op = tf.assign(vars_6, vars_3)
            copy_ops.append(copy_op)
        self.session.run(copy_ops)

        coef3 = self.session.run(self.coef3,
                                 feed_dict={self.A: A, self.X: X, self.R: R, self.S: S,
                                            self.A2: A2, self.X2: X2, self.R2: R2, self.S2: S2})
        alpha = max(0.4 - (3 - 1) / 10 * 0.1, 0.1)
        coef_ans = coef3
        commonZ = thrC(coef_ans, alpha)
        y_x, _ = post_proC(commonZ, self.args.cluster, 20, 3.5)
        s2_label_subjs = np.array(y_x)
        s2_label_subjs = s2_label_subjs - s2_label_subjs.min() + 1
        s2_label_subjs = np.squeeze(s2_label_subjs)
        one_hot_Label = get_one_hot_Label(s2_label_subjs, self.args.cluster)
        s2_Q = form_structure_matrix(s2_label_subjs, self.args.cluster)
        s2_Theta = form_Theta(s2_Q)
        Y = y_x
        Y = Y.reshape(self.args.n_sample, 1)

        print("-------------------------------------------------------------")
        print("Initial Clustering Results: ")
        print("acc: {:.8f}\t\tnmi: {:.8f}\t\tf_score: {:.8f}\t\tari: {:.8f}".
              format(cluster_acc(y_true, y_x - 1), nmi(y_true, y_x - 1), f_score(y_true, y_x - 1),
                     ari(y_true, y_x - 1)))
        print("-------------------------------------------------------------")

        for epoch in range(self.args.n_epochs):
            loss, _, ft_loss, st_loss, SE_loss, C_Regular, consistent_loss, cl_loss, Cq_loss, coef3 = self.session.run(
                [self.loss, self.train_op, self.ft_loss, self.st_loss, self.SE_loss, self.C_Regular, self.consistent_loss, self.cl_loss, self.Cq_loss, self.coef3],
                feed_dict={self.A: A, self.X: X, self.R: R, self.S: S, self.p: one_hot_Label, self.y_pred: Y - 1,
                           self.A2: A2, self.X2: X2, self.R2: R2, self.S2: S2, self.Theta: s2_Theta})

            # if epoch % 80 == 0:
            coef_ans = coef3
            commonZ = thrC(coef_ans, alpha)
            y_x, _ = post_proC(commonZ, self.args.cluster, 20, 3.5)
            print("Epoch--{}:\t\tloss: {:.8f}\t\tacc: {:.8f}\t\tnmi: {:.8f}\t\tf1: {:.8f}\t\tari: {:.8f}".
                  format(epoch, loss, cluster_acc(y_true, y_x - 1), nmi(y_true, y_x - 1),
                         f_score(y_true, y_x - 1), ari(y_true, y_x - 1)))
            print("Epoch--{}:\t\tft_loss: {:.8f}\t\tst_loss: {:.8f}\t\tSE_loss: {:.8f}\t\tC_Regular: {:.8f}"
                  "\t\tcl_loss: {:.8f}\t\tCq_loss: {:.8f}\t\tconsistent_loss: {:.8f}".
                  format(epoch, ft_loss, st_loss, SE_loss, C_Regular, cl_loss, Cq_loss, consistent_loss))

            if epoch % 10 == 0:
                # coef_ans = coef3
                # commonZ = thrC(coef_ans, alpha)
                # y_x, _ = post_proC(commonZ, self.args.cluster, 20, 3.5)
                Y = y_x
                Y = Y.reshape(self.args.n_sample, 1)
                s2_label_subjs = np.array(Y)
                s2_label_subjs = s2_label_subjs - s2_label_subjs.min() + 1
                s2_label_subjs = np.squeeze(s2_label_subjs)
                one_hot_Label = get_one_hot_Label(s2_label_subjs, self.args.cluster)
                s2_Q = form_structure_matrix(s2_label_subjs, self.args.cluster)
                s2_Theta = form_Theta(s2_Q)
