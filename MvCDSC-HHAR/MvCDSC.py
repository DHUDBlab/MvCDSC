import torch.utils.data
from torch import nn
import numpy as np
import torch
import torch.nn.functional as F
import gc
import os

device = torch.device("cuda:1")
device2 = torch.device("cuda:0")

# class SUREfcCaltech(nn.Module):
#     def __init__(self, node_dim1, node_dim2):
#         super(SUREfcCaltech, self).__init__()
#         num_fea = 512
#         self.encoder = nn.Sequential(
#             nn.Linear(node_dim1, node_dim2),
#             nn.BatchNorm1d(node_dim2),
#             nn.ReLU(True)
#             #
#             # nn.Linear(1024, 1024),
#             # nn.BatchNorm1d(1024),
#             # nn.ReLU(True),
#
#             # nn.Linear(1024, num_fea),
#             # nn.BatchNorm1d(num_fea),
#             # nn.ReLU(True)
#         )
#
#     def forward(self, X):
#         h0 = self.encoder(X)
#         h0 = F.normalize(h0, dim=1)
#         return h0


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, args):
        super(GCNLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.args = args
        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(in_features, out_features).type(
            torch.cuda.FloatTensor if args.cuda else torch.FloatTensor), gain=np.sqrt(2.0)),
            requires_grad=True)

    def forward(self, x, sp_adj, is_sparse_input=False):
        if is_sparse_input:
            h = torch.spmm(x, self.W)
        else:
            h = torch.mm(x, self.W)
        h_prime = torch.spmm(sp_adj, h)
        return F.elu(h_prime)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args
        self.lambda_1 = self.args.lambda_1
        self.lambda_2 = self.args.lambda_2
        self.lambda_3 = self.args.lambda_3
        self.lambda_4 = self.args.lambda_4
        self.lambda_5 = self.args.lambda_5

        self.C = {}
        self.weight = nn.Parameter(1.0e-4 * torch.ones((args.n_nodes, args.n_nodes)))

        self.C2 = {}
        self.weight2 = nn.Parameter(1.0e-4 * torch.ones((args.n_nodes, args.n_nodes)))

        self.C31 = {}
        self.C32 = {}
        self.weight31 = nn.Parameter(1.0e-4 * torch.ones((args.n_nodes, args.n_nodes)))
        self.weight32 = nn.Parameter(1.0e-4 * torch.ones((args.n_nodes, args.n_nodes)))

        self.enc_layer11 = GCNLayer(args.feat1, args.hidden1, args.dropout, args)
        self.enc_layer12 = GCNLayer(args.hidden1, args.hidden2, args.dropout, args)
        self.dec_layer11 = GCNLayer(args.hidden2, args.hidden1, args.dropout, args)
        self.dec_layer12 = GCNLayer(args.hidden1, args.feat1, args.dropout, args)

        self.enc_layer21 = GCNLayer(args.feat2, args.hidden1, args.dropout, args)
        self.enc_layer22 = GCNLayer(args.hidden1, args.hidden2, args.dropout, args)
        self.dec_layer21 = GCNLayer(args.hidden2, args.hidden1, args.dropout, args)
        self.dec_layer22 = GCNLayer(args.hidden1, args.feat2, args.dropout, args)

        self.enc_layer31 = GCNLayer(args.hidden2, args.hidden3, args.dropout, args)
        # self.enc_layer32 = GCNLayer(args.hidden3, args.hidden4, args.dropout, args)
        self.dec_layer31 = GCNLayer(args.hidden3, args.hidden2, args.dropout, args)
        # self.dec_layer32 = GCNLayer(args.hidden3, args.hidden2, args.dropout, args)

        self.n_cluster = args.n_cluster

    def forward(self, X, A, S, R, X2, A2, S2, R2, y_pred, Theta):
        coef = self.weight - torch.diag(torch.diag(self.weight))
        coef2 = self.weight2 - torch.diag(torch.diag(self.weight2))
        coef31 = self.weight31 - torch.diag(torch.diag(self.weight31))
        coef32 = self.weight32 - torch.diag(torch.diag(self.weight32))

        # X = F.dropout(X, self.args.dropout, training=self.training)
        # Encoder1
        H = self.enc_layer11(X, A)
        # H = F.dropout(H, self.args.dropout, training=self.training)
        H = self.enc_layer12(H, A)
        # Final node representations
        self.H = H
        self.HC = torch.matmul(coef, H)
        H = self.HC

        # Decoder1
        H = self.dec_layer11(H, A)
        X_ = self.dec_layer12(H, A)
        # mark
        self.Z = self.H

        # Encoder2
        # X2 = F.dropout(X2, self.args.dropout, training=self.training)
        H2 = self.enc_layer21(X2, A2)
        # H2 = F.dropout(H2, self.args.dropout, training=self.training)
        H2 = self.enc_layer22(H2, A2)
        # Final node representations
        self.H2 = H2
        self.HC2 = torch.matmul(coef2, H2)
        H2 = self.HC2

        # Decoder2
        H2 = self.dec_layer21(H2, A2)
        X2_ = self.dec_layer22(H2, A2)
        # mark
        self.Z2 = self.H2

        # Encoder3-1
        H31 = self.Z
        Z1 = self.Z
        # H31 = F.dropout(H31, self.args.dropout, training=self.training)
        H31 = self.enc_layer31(H31, A)
        # H31 = F.dropout(H31, self.args.dropout, training=self.training)
        # H31 = self.enc_layer32(H31, A)
        self.H31 = H31
        self.HC31 = torch.matmul(coef31, H31)
        H31 = self.HC31

        # Decoder3-1
        H31 = self.dec_layer31(H31, A)
        # H31 = self.dec_layer32(H31, A)
        Z1_ = H31
        self.Z31 = self.H31

        # Encoder3-2
        H32 = self.Z2
        Z2 = self.Z2
        # H32 = F.dropout(H32, self.args.dropout, training=self.training)
        H32 = self.enc_layer31(H32, A2)
        # H32 = F.dropout(H32, self.args.dropout, training=self.training)
        # H32 = self.enc_layer32(H32, A2)
        self.H32 = H32
        self.HC32 = torch.matmul(coef32, H32)
        H32 = self.HC32

        # Decoder3-2
        H32 = self.dec_layer31(H32, A2)
        # H32 = self.dec_layer32(H32, A2)
        Z2_ = H32
        self.Z32 = self.H32

        # The reconstruction loss of node features
        self.ft_loss1 = torch.mean((X - X_) ** 2)
        self.ft_loss2 = torch.mean((X2 - X2_) ** 2)
        self.ft_loss3 = torch.mean((Z1 - Z1_) ** 2) + torch.mean((Z2 - Z2_) ** 2)
        self.ft_loss = self.ft_loss1 + self.ft_loss2 + self.ft_loss3
        # The reconstruction loss of the graph structure
        self.S_emb = self.H[S]
        self.R_emb = self.H[R]

        self.S_emb2 = self.H2[S2]
        self.R_emb2 = self.H2[R2]

        self.S_emb31 = self.H31[S]
        self.R_emb31 = self.H31[R]

        self.S_emb32 = self.H32[S2]
        self.R_emb32 = self.H32[R2]

        self.st_loss1 = -torch.log(torch.sigmoid(torch.sum(self.S_emb * self.R_emb, dim=-1)))
        self.st_loss2 = -torch.log(torch.sigmoid(torch.sum(self.S_emb2 * self.R_emb2, dim=-1)))
        self.st_loss31 = -torch.log(torch.sigmoid(torch.sum(self.S_emb31 * self.R_emb31, dim=-1)))
        self.st_loss32 = -torch.log(torch.sigmoid(torch.sum(self.S_emb32 * self.R_emb32, dim=-1)))
        self.st_loss = torch.sum(self.st_loss1) + torch.sum(self.st_loss2) + torch.sum(self.st_loss31) + torch.sum(self.st_loss32)

        # The loss of self-expression and C-penalty term
        self.SE_loss1 = 0.5 * torch.mean((self.H - self.HC) ** 2)
        self.SE_loss2 = 0.5 * torch.mean((self.H2 - self.HC2) ** 2)
        self.SE_loss3 = 0.5 * torch.mean((self.H31 - self.HC31) ** 2) + 0.5 * torch.mean(
            (self.H32 - self.HC32) ** 2)
        self.SE_loss = self.SE_loss1 + self.SE_loss2 + self.SE_loss3

        coef = coef.to(device2)
        coef2 = coef2.to(device2)
        coef31 = coef31.to(device2)
        coef32 = coef32.to(device2)
        self.C_Regular1 = torch.sum(torch.abs(coef) ** 1.0)
        self.C_Regular2 = torch.sum(torch.abs(coef2) ** 1.0)
        self.C_Regular31 = torch.sum(torch.abs(coef31) ** 1.0)
        self.C_Regular32 = torch.sum(torch.abs(coef32) ** 1.0)
        self.C_Regular = self.C_Regular1 + self.C_Regular2 + self.C_Regular31 + self.C_Regular32

        # Contrastive Loss
        self.cl_loss = self.constrastive_loss(coef31, coef32, self.args.n_nodes, y_pred)
        # Final coef
        self.coef3 = 0.7 * coef31 + 0.3 * coef32
        # Cpq Loss
        self.Cq_loss = torch.sum(torch.pow(torch.abs(self.coef3.t().cpu().detach() * Theta), 1.0))
        # Consistent Loss
        self.consistent_loss = torch.sum((self.coef3 - coef) ** 2) + torch.sum(
            (self.coef3 - coef2) ** 2)

        coef = coef.to(device)
        coef2 = coef2.to(device)
        coef31 = coef31.to(device)
        coef32 = coef32.to(device)
        self.C_Regular = self.C_Regular.to(device)
        self.consistent_loss = self.consistent_loss.to(device)

        # Total loss
        self.loss = self.ft_loss + self.lambda_1 * self.st_loss + self.SE_loss + self.lambda_2 * self.C_Regular \
                     + self.lambda_3 * self.cl_loss \
                     + self.lambda_4 * self.Cq_loss \
                     + self.lambda_5 * self.consistent_loss

        return self.loss, self.ft_loss, self.st_loss, self.SE_loss, self.C_Regular, self.consistent_loss, self.cl_loss, self.Cq_loss, self.coef3

    def constrastive_loss(self, z_i, z_j, batch_size, y_pred, temperature=1.0):
        negative_mask1 = np.ones(shape=(batch_size, batch_size)).astype('float32')
        negative_mask2 = np.ones(shape=(batch_size, batch_size)).astype('float32')
        temp_mask = (y_pred == y_pred.transpose(1, 0))
        temp_mask = temp_mask.astype(float)
        negative_mask1 = negative_mask1 - temp_mask
        negative_mask2 = negative_mask2 - temp_mask
        negative_mask1 = torch.from_numpy(negative_mask1)
        negative_mask2 = torch.from_numpy(negative_mask2)
        negative_mask = torch.cat([negative_mask1, negative_mask2], dim=1)
        negative_mask = negative_mask.to(device2)

        zis = F.normalize(z_i, p=2, dim=1)
        zjs = F.normalize(z_j, p=2, dim=1)
        l_pos = self.dot_simililarity_dim1(zis, zjs)
        l_pos = l_pos.view(batch_size, 1)
        l_pos /= temperature
        l_pos = torch.exp(l_pos)
        l_pos = l_pos.to(device2)

        negatives = torch.cat([zjs, zis], dim=0)  # (2,3)+(4,3)=(6,3)

        loss = 0
        for positives in [zis, zjs]:
            l_neg = self.dot_simililarity_dim2(positives, negatives)  # 一个实例和俩个视图的所有实例的点乘，N*2N
            l_neg /= temperature
            l_neg.exp_()
            l_neg = l_neg.to(device2)

            exp_logits = l_neg * negative_mask
            sum_exp_logits = exp_logits.sum(dim=1)
            sum_exp_logits = sum_exp_logits.view(batch_size, 1)
            ans = (-1 * torch.log(l_pos / (l_pos + sum_exp_logits))).sum().item()
            loss += ans

        loss = loss / (2 * batch_size)
        return loss

    def dot_simililarity_dim1(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (N, C, 1)
        # v shape: (N, 1, 1)
        v = torch.matmul(x.unsqueeze(1), y.unsqueeze(-1))
        return v

    def dot_simililarity_dim2(self, x, y):
        x_expanded = x.unsqueeze(1).to(device2)
        y_transposed = y.t().to(device2)

        v = (x_expanded @ y_transposed).squeeze(1)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v


class IndividualMLPEncoder(nn.Module):
    def __init__(self, n_fts, args):
        super(IndividualMLPEncoder, self).__init__()
        self.encoder1 = GCNLayer(n_fts, args.hidden1, args.dropout, args)
        self.encoder2 = GCNLayer(args.hidden1, args.hidden2, args.dropout, args)
        self.decoder1 = GCNLayer(args.hidden2, args.hidden1, args.dropout, args)
        self.decoder2 = GCNLayer(args.hidden1, n_fts, args.dropout, args)

        self.C = {}
        self.weight = nn.Parameter(1.0e-4 * torch.ones((args.n_nodes, args.n_nodes)))

    def forward(self, X, A, S, R, args):
        X = F.dropout(X, args.dropout, training=self.training)
        H = self.encoder1(X, A)
        H = F.dropout(H, args.dropout, training=self.training)
        H = self.encoder2(H, A)

        self.H = H
        coef = self.weight - torch.diag(torch.diag(self.weight))
        self.HC = torch.matmul(coef, H)

        H = self.HC
        H = self.decoder1(H, A)
        X_ = self.decoder2(H, A)

        self.ft_loss = torch.mean((X - X_) ** 2)

        self.S_emb = self.H[S]
        self.R_emb = self.H[R]
        self.st_loss1 = -torch.log(torch.sigmoid(torch.sum(self.S_emb * self.R_emb, dim=-1)))
        self.st_loss = torch.sum(self.st_loss1)

        self.SE_loss = 0.5 * torch.mean((self.H - self.HC) ** 2)
        self.C_Regular = torch.sum(torch.pow(torch.abs(coef), 1.0))
        # Total loss
        self.loss = self.ft_loss + args.lambda_1 * self.st_loss + self.SE_loss + self.C_Regular

        return self.H, self.loss, self.ft_loss, self.st_loss, self.SE_loss, self.C_Regular


class CommonMLPEncoder(nn.Module):
    def __init__(self, args):
        super(CommonMLPEncoder, self).__init__()
        self.encoder1 = GCNLayer(args.hidden2, args.hidden3, args.dropout, args)
        # self.encoder2 = GCNLayer(args.hidden3, args.hidden4, args.dropout, args)
        self.decoder1 = GCNLayer(args.hidden3, args.hidden2, args.dropout, args)
        # self.decoder2 = GCNLayer(args.hidden3, args.hidden2, args.dropout, args)

        self.C1 = {}
        self.weight1 = nn.Parameter(1.0e-4 * torch.ones((args.n_nodes, args.n_nodes)))

        self.C2 = {}
        self.weight2 = nn.Parameter(1.0e-4 * torch.ones((args.n_nodes, args.n_nodes)))

    def forward(self, H1, A1, S, R, H2, A2, S2, R2, args):
        H1 = F.dropout(H1, args.dropout, training=self.training)
        Z1 = self.encoder1(H1, A1)
        # Z1 = F.dropout(Z1, args.dropout, training=self.training)
        # Z1 = self.encoder2(Z1, A1)
        self.Z1 = Z1

        coef1 = self.weight1 - torch.diag(torch.diag(self.weight1))
        self.ZC1 = torch.matmul(coef1, Z1)
        Z1 = self.ZC1

        H1_ = self.decoder1(Z1, A1)
        # H1_ = self.decoder2(H1_, A1)

        H2 = F.dropout(H2, args.dropout, training=self.training)
        Z2 = self.encoder1(H2, A2)
        # Z2 = F.dropout(Z2, args.dropout, training=self.training)
        # Z2 = self.encoder2(Z2, A2)
        self.Z2 = Z2

        coef2 = self.weight2 - torch.diag(torch.diag(self.weight2))
        self.ZC2 = torch.matmul(coef2, Z2)
        Z2 = self.ZC2

        H2_ = self.decoder1(Z2, A2)
        # H2_ = self.decoder2(H2_, A2)

        self.ft_loss = torch.mean((H1 - H1_) ** 2) + torch.mean((H2 - H2_) ** 2)

        self.S_emb = self.Z1[S]
        self.R_emb = self.Z1[R]
        self.st_loss1 = -torch.log(torch.sigmoid(torch.sum(self.S_emb * self.R_emb, dim=-1)))
        self.S_emb2 = self.Z2[S2]
        self.R_emb2 = self.Z2[R2]
        self.st_loss2 = -torch.log(torch.sigmoid(torch.sum(self.S_emb2 * self.R_emb2, dim=-1)))
        self.st_loss = torch.sum(self.st_loss1) + torch.sum(self.st_loss2)

        self.SE_loss = 0.5 * torch.mean((self.Z1 - self.ZC1) ** 2) + 0.5 * torch.mean((self.Z2 - self.ZC2) ** 2)
        self.C_Regular = torch.sum(torch.pow(torch.abs(coef1), 1.0)) + torch.sum(torch.pow(torch.abs(coef2), 1.0))
        # Total loss
        self.loss = self.ft_loss + args.lambda_1 * self.st_loss + self.SE_loss + self.C_Regular

        return coef1, coef2, self.loss, self.ft_loss, self.st_loss, self.SE_loss, self.C_Regular