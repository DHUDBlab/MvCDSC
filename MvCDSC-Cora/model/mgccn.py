import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans


def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


class MGCCN():
    def __init__(self, args):
        self.args = args
        self.lambda_1 = self.args.lambda_1
        self.lambda_2 = self.args.lambda_2
        self.lambda_3 = self.args.lambda_3
        self.lambda_4 = self.args.lambda_4
        self.lambda_5 = self.args.lambda_5
        self.n_layers1 = len(self.args.hidden_dims1) - 1
        self.n_layers2 = len(self.args.hidden_dims2) - 1
        self.n_layers3 = len(self.args.hidden_dims3) - 1
        self.W, self.v = self.define_weights(self.args.hidden_dims1)
        self.C = {}
        self.weight = tf.Variable(1.0e-4 * tf.ones(shape=(args.n_sample, args.n_sample)), name="DP1weight")
        self.coef = self.weight - tf.matrix_diag(tf.diag_part(self.weight))  # diag_part返回张量的对角线部分
        self.W2, self.v2 = self.define_weights2(self.args.hidden_dims2)
        self.C2 = {}
        self.weight2 = tf.Variable(1.0e-4 * tf.ones(shape=(args.n_sample, args.n_sample)), name="DP2weight")
        self.coef2 = self.weight2 - tf.matrix_diag(tf.diag_part(self.weight2))
        self.W3, self.v3 = self.define_weights3(self.args.hidden_dims3)
        self.C31 = {}
        self.C32 = {}
        self.weight31 = tf.Variable(1.0e-4 * tf.ones(shape=(args.n_sample, args.n_sample)), name="DP31weight")
        self.coef31 = self.weight31 - tf.matrix_diag(tf.diag_part(self.weight31))
        self.weight32 = tf.Variable(1.0e-4 * tf.ones(shape=(args.n_sample, args.n_sample)), name="DP32weight")
        self.coef32 = self.weight32 - tf.matrix_diag(tf.diag_part(self.weight32))
        self.n_cluster = self.args.cluster
        self.input_batch_size = self.args.n_sample
        self.alpha = self.args.alpha

    def __call__(self, A, X, R, S, p, y_pred, A2, X2, R2, S2, Theta):
        # Encoder1
        H = X
        for layer in range(self.n_layers1):
            H = self.__encoder(A, H, layer)
        # Final node representations
        self.H = H
        self.HC = tf.matmul(self.coef, H)
        H = self.HC

        # Decoder1
        for layer in range(self.n_layers1 - 1, -1, -1):
            H = self.__decoder(H, layer)
        X_ = H
        self.Z = self.H

        # Encoder2
        H2 = X2
        for layer in range(self.n_layers2):
            H2 = self.__encoder2(A2, H2, layer)
        # Final node representations
        self.H2 = H2
        self.HC2 = tf.matmul(self.coef2, H2)
        H2 = self.HC2

        # Decoder2
        for layer in range(self.n_layers2 - 1, -1, -1):
            H2 = self.__decoder2(H2, layer)
        X_2 = H2
        self.Z2 = self.H2

        # Encoder3-1
        H31 = self.Z
        Z1 = self.Z
        for layer in range(self.n_layers3):
            H31 = self.__encoder31(A, H31, layer)
        self.H31 = H31
        self.HC31 = tf.matmul(self.coef31, H31)
        H31 = self.HC31

        # Decoder3-1
        for layer in range(self.n_layers3 - 1, -1, -1):
            H31 = self.__decoder31(H31, layer)
        Z1_ = H31
        self.Z31 = self.H31

        # Encoder3-2
        H32 = self.Z2
        Z2 = self.Z2
        for layer in range(self.n_layers3):
            H32 = self.__encoder32(A, H32, layer)
        self.H32 = H32
        self.HC32 = tf.matmul(self.coef32, H32)
        H32 = self.HC32

        # Decoder3-2
        for layer in range(self.n_layers3 - 1, -1, -1):
            H32 = self.__decoder32(H32, layer)
        Z2_ = H32
        self.Z32 = self.H32

        # The reconstruction loss of node features
        self.ft_loss1 = tf.reduce_mean((X - X_) ** 2)
        self.ft_loss2 = tf.reduce_mean((X2 - X_2) ** 2)
        self.ft_loss3 = tf.reduce_mean((Z1 - Z1_) ** 2) + tf.reduce_mean((Z2 - Z2_) ** 2)
        self.ft_loss = self.ft_loss1 + self.ft_loss2 + self.ft_loss3
        # The reconstruction loss of the graph structure
        self.S_emb = tf.nn.embedding_lookup(self.H, S)  # 选取一个张量里面索引对应的元素
        self.R_emb = tf.nn.embedding_lookup(self.H, R)
        self.S_emb2 = tf.nn.embedding_lookup(self.H2, S2)
        self.R_emb2 = tf.nn.embedding_lookup(self.H2, R2)
        self.S_emb31 = tf.nn.embedding_lookup(self.H31, S)
        self.R_emb31 = tf.nn.embedding_lookup(self.H31, R)
        self.S_emb32 = tf.nn.embedding_lookup(self.H32, S)
        self.R_emb32 = tf.nn.embedding_lookup(self.H32, R)
        self.st_loss1 = -tf.log(tf.sigmoid(tf.reduce_sum(self.S_emb * self.R_emb, axis=-1)))  # reduce_sum()是按一定方式计算张量中的元素之和
        self.st_loss2 = -tf.log(tf.sigmoid(tf.reduce_sum(self.S_emb2 * self.R_emb2, axis=-1)))
        self.st_loss31 = -tf.log(tf.sigmoid(tf.reduce_sum(self.S_emb31 * self.R_emb31, axis=-1)))
        self.st_loss32 = -tf.log(tf.sigmoid(tf.reduce_sum(self.S_emb32 * self.R_emb32, axis=-1)))
        self.st_loss = tf.reduce_sum(self.st_loss1) + tf.reduce_sum(self.st_loss2) + tf.reduce_sum(self.st_loss31) + tf.reduce_sum(self.st_loss32)
        # The loss of self-expression and C-penalty term
        self.SE_loss1 = 0.5 * tf.reduce_mean((self.H - self.HC) ** 2)
        self.SE_loss2 = 0.5 * tf.reduce_mean((self.H2 - self.HC2) ** 2)
        self.SE_loss3 = 0.5 * tf.reduce_mean((self.H31 - self.HC31) ** 2) + 0.5 * tf.reduce_mean((self.H32 - self.HC32) ** 2)
        self.SE_loss = self.SE_loss1 + self.SE_loss2 + self.SE_loss3
        self.C_Regular1 = tf.reduce_sum(tf.pow(tf.abs(self.coef), 1.0))
        self.C_Regular2 = tf.reduce_sum(tf.pow(tf.abs(self.coef2), 1.0))
        self.C_Regular31 = tf.reduce_sum(tf.pow(tf.abs(self.coef31), 1.0))
        self.C_Regular32 = tf.reduce_sum(tf.pow(tf.abs(self.coef32), 1.0))
        self.C_Regular = self.C_Regular1 + self.C_Regular2 + self.C_Regular31 + self.C_Regular32
        # Contrastive Loss
        self.cl_loss = self._constrastive_loss(self.coef31, self.coef32, self.args.n_sample, y_pred)
        # Final coef
        self.coef3 = 0.7 * self.coef31 + 0.3 * self.coef32
        # Cpq Loss
        self.Cq_loss = tf.reduce_sum(tf.pow(tf.abs(tf.transpose(self.coef3) * Theta), 1.0))
        # Consistent Loss
        self.consistent_loss = tf.reduce_sum((self.coef3 - self.coef) ** 2) + tf.reduce_sum((self.coef3 - self.coef2) ** 2)
        # Total loss
        self.loss = self.ft_loss + self.lambda_1 * self.st_loss + self.SE_loss + self.lambda_2 * self.C_Regular + self.lambda_3 * self.cl_loss \
            + self.lambda_4 * self.Cq_loss + self.lambda_5 * self.consistent_loss

        return self.loss, self.ft_loss, self.st_loss, self.SE_loss, self.C_Regular, self.consistent_loss, self.cl_loss, self.Cq_loss, \
            self.H, self.C, self.H2, self.C2, self.Z, self.Z2, self.coef3

    def __encoder(self, A, H, layer):
        H = tf.matmul(H, self.W[layer])
        self.C[layer] = self.graph_attention_layer(A, H, self.v[layer], layer)
        return tf.sparse_tensor_dense_matmul(self.C[layer], H)

    def __decoder(self, H, layer):
        H = tf.matmul(H, self.W[layer], transpose_b=True)
        return tf.sparse_tensor_dense_matmul(self.C[layer], H)

    def __encoder2(self, A, H, layer):
        H = tf.matmul(H, self.W2[layer])
        self.C2[layer] = self.graph_attention_layer2(A, H, self.v2[layer], layer)
        return tf.sparse_tensor_dense_matmul(self.C2[layer], H)

    def __decoder2(self, H, layer):
        H = tf.matmul(H, self.W2[layer], transpose_b=True)
        return tf.sparse_tensor_dense_matmul(self.C2[layer], H)

    def __encoder31(self, A, H, layer):
        H = tf.matmul(H, self.W3[layer])
        self.C31[layer] = self.graph_attention_layer3(A, H, self.v3[layer], layer)
        return tf.sparse_tensor_dense_matmul(self.C31[layer], H)

    def __decoder31(self, H, layer):
        H = tf.matmul(H, self.W3[layer], transpose_b=True)
        return tf.sparse_tensor_dense_matmul(self.C31[layer], H)

    def __encoder32(self, A, H, layer):
        H = tf.matmul(H, self.W3[layer])
        self.C32[layer] = self.graph_attention_layer3(A, H, self.v3[layer], layer)
        return tf.sparse_tensor_dense_matmul(self.C32[layer], H)

    def __decoder32(self, H, layer):
        H = tf.matmul(H, self.W3[layer], transpose_b=True)
        return tf.sparse_tensor_dense_matmul(self.C32[layer], H)

    def define_weights(self, hidden_dims):
        W = {}
        for i in range(self.n_layers1):
            W[i] = tf.get_variable("DP1W%s" % i, shape=(hidden_dims[i], hidden_dims[i + 1]))

        Ws_att = {}
        for i in range(self.n_layers1):
            v = {}
            v[0] = tf.get_variable("DP1v%s_0" % i, shape=(hidden_dims[i + 1], 1))
            v[1] = tf.get_variable("DP1v%s_1" % i, shape=(hidden_dims[i + 1], 1))
            Ws_att[i] = v

        return W, Ws_att

    def define_weights2(self, hidden_dims):
        W2 = {}
        for i in range(self.n_layers2):
            W2[i] = tf.get_variable("DP2W2%s" % i, shape=(hidden_dims[i], hidden_dims[i + 1]))

        Ws_att2 = {}
        for i in range(self.n_layers2):
            v2 = {}
            v2[0] = tf.get_variable("DP2v2%s_0" % i, shape=(hidden_dims[i + 1], 1))
            v2[1] = tf.get_variable("DP2v2%s_1" % i, shape=(hidden_dims[i + 1], 1))
            Ws_att2[i] = v2

        return W2, Ws_att2

    def define_weights3(self, hidden_dims):
        W3 = {}
        for i in range(self.n_layers3):
            W3[i] = tf.get_variable("DP3W3%s" % i, shape=(hidden_dims[i], hidden_dims[i + 1]))

        Ws_att3 = {}
        for i in range(self.n_layers3):
            v3 = {}
            v3[0] = tf.get_variable("DP3v3%s_0" % i, shape=(hidden_dims[i + 1], 1))
            v3[1] = tf.get_variable("DP3v3%s_1" % i, shape=(hidden_dims[i + 1], 1))
            Ws_att3[i] = v3

        return W3, Ws_att3

    def graph_attention_layer(self, A, M, v, layer):
        with tf.variable_scope("layer_%s" % layer):
            f1 = tf.matmul(M, v[0])
            f1 = A * f1
            f2 = tf.matmul(M, v[1])
            f2 = A * tf.transpose(f2, [1, 0])
            logits = tf.sparse_add(f1, f2)
            unnormalized_attentions = tf.SparseTensor(indices=logits.indices,
                                                      values=tf.nn.sigmoid(logits.values),
                                                      dense_shape=logits.dense_shape)
            attentions = tf.sparse_softmax(unnormalized_attentions)

            attentions = tf.SparseTensor(indices=attentions.indices,
                                         values=attentions.values,
                                         dense_shape=attentions.dense_shape)

            return attentions

    def graph_attention_layer2(self, A, M, v, layer):
        with tf.variable_scope("layer_%s" % layer):
            f1 = tf.matmul(M, v[0])
            f1 = A * f1
            f2 = tf.matmul(M, v[1])
            f2 = A * tf.transpose(f2, [1, 0])
            logits = tf.sparse_add(f1, f2)
            unnormalized_attentions = tf.SparseTensor(indices=logits.indices,
                                                      values=tf.nn.sigmoid(logits.values),
                                                      dense_shape=logits.dense_shape)
            attentions = tf.sparse_softmax(unnormalized_attentions)

            attentions = tf.SparseTensor(indices=attentions.indices,
                                         values=attentions.values,
                                         dense_shape=attentions.dense_shape)

            return attentions

    def graph_attention_layer3(self, A, M, v, layer):
        with tf.variable_scope("layer_%s" % layer):
            f1 = tf.matmul(M, v[0])
            f1 = A * f1
            f2 = tf.matmul(M, v[1])
            f2 = A * tf.transpose(f2, [1, 0])
            logits = tf.sparse_add(f1, f2)
            unnormalized_attentions = tf.SparseTensor(indices=logits.indices,
                                                      values=tf.nn.sigmoid(logits.values),
                                                      dense_shape=logits.dense_shape)
            attentions = tf.sparse_softmax(unnormalized_attentions)

            attentions = tf.SparseTensor(indices=attentions.indices,
                                         values=attentions.values,
                                         dense_shape=attentions.dense_shape)

            return attentions

    def _soft_assignment(self, embeddings, cluster_centers):
        """Implemented a soft assignment as the  probability of assigning sample i to cluster j.

        Args:
            embeddings: (num_points, dim)
            cluster_centers: (num_cluster, dim)

        Return:
            q_i_j: (num_points, num_cluster)
        """

        def _pairwise_euclidean_distance(a, b):
            p1 = tf.matmul(
                tf.expand_dims(tf.reduce_sum(tf.square(a), 1), 1),
                tf.ones(shape=(1, self.n_cluster))
            )
            p2 = tf.transpose(tf.matmul(
                tf.reshape(tf.reduce_sum(tf.square(b), 1), shape=[-1, 1]),
                tf.ones(shape=(self.input_batch_size, 1)),
                transpose_b=True
            ))
            res = tf.sqrt(tf.add(p1, p2) - 2 * tf.matmul(a, b, transpose_b=True))
            return res

        # print("embeddings.shape", embeddings.shape)
        dist = _pairwise_euclidean_distance(embeddings, cluster_centers)
        q = 1.0 / (1.0 + dist ** 2 / self.alpha) ** ((self.alpha + 1.0) / 2.0)
        q = (q / tf.reduce_sum(q, axis=1, keepdims=True))
        return q

    def target_distribution(self, q):
        p = q ** 2 / q.sum(axis=0)
        p = p / p.sum(axis=1, keepdims=True)
        return p

    def _self_supervised_clustering(self, target, pred):
        return tf.reduce_mean((target - pred) ** 2)

    def _constrastive_loss(self, z_i, z_j, batch_size, y_pred, temperature=1.0):
        negative_mask1 = np.ones(shape=(batch_size, batch_size)).astype('float32')
        negative_mask2 = np.ones(shape=(batch_size, batch_size)).astype('float32')
        temp_mask = tf.equal(y_pred, tf.transpose(y_pred, [1, 0]))
        temp_mask = tf.cast(temp_mask, dtype=tf.float32)
        negative_mask1 = negative_mask1 - temp_mask
        negative_mask2 = negative_mask2 - temp_mask
        negative_mask = tf.concat([negative_mask1, negative_mask2], axis=1)

        zis = tf.nn.l2_normalize(z_i, axis=1)
        zjs = tf.nn.l2_normalize(z_j, axis=1)
        l_pos = self._dot_simililarity_dim1(zis, zjs)
        l_pos = tf.reshape(l_pos, (batch_size, 1))
        l_pos /= temperature
        l_pos = tf.exp(l_pos)

        negatives = tf.concat([zjs, zis], axis=0)  # (2,3)+(4,3)=(6,3)

        loss = 0
        for positives in [zis, zjs]:
            l_neg = self._dot_simililarity_dim2(positives, negatives)  # 一个实例和俩个视图的所有实例的点乘，N*2N
            l_neg /= temperature
            exp_logits = tf.exp(l_neg) * negative_mask
            sum_exp_logits = tf.reduce_sum(exp_logits, axis=1)
            sum_exp_logits = tf.reshape(sum_exp_logits, (batch_size, 1))
            ans = tf.reduce_sum(-1 * tf.log(l_pos / (l_pos + sum_exp_logits)))
            loss += ans

        loss = loss / (2 * batch_size)
        return loss

    def get_negative_mask(self, batch_size):
        # return a mask that removes the similarity score of equal/similar images.
        # this function ensures that only distinct pair of images get their similarity scores
        # passed as negative examples
        negative_mask = np.ones((batch_size, 2 * batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
            negative_mask[i, i + batch_size] = 0
        return tf.constant(negative_mask)

    def _cosine_simililarity_dim1(self, x, y):
        cosine_sim_1d = tf.keras.losses.CosineSimilarity(axis=1, reduction=tf.keras.losses.Reduction.NONE)
        v = cosine_sim_1d(x, y)
        return v

    def _cosine_simililarity_dim2(self, x, y):
        cosine_sim_2d = tf.keras.losses.CosineSimilarity(axis=2, reduction=tf.keras.losses.Reduction.NONE)
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = cosine_sim_2d(tf.expand_dims(x, 1), tf.expand_dims(y, 0))
        return v

    def _dot_simililarity_dim1(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (N, C, 1)
        # v shape: (N, 1, 1)
        v = tf.matmul(tf.expand_dims(x, 1), tf.expand_dims(y, 2))
        return v

    def _dot_simililarity_dim2(self, x, y):
        v = tf.tensordot(tf.expand_dims(x, 1), tf.expand_dims(tf.transpose(y), 0), axes=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def __cosine_similarity(self, z, z2):
        z = tf.nn.l2_normalize(z, axis=1)
        z2 = tf.nn.l2_normalize(z2, axis=1)
        return tf.reduce_mean(tf.reduce_sum(-(z * z2), axis=1))

    def HSIC(self, c_v, c_w):
        N = tf.shape(c_v)[0]
        H = tf.ones((N, N)) * tf.cast((1 / N), tf.float32) * (-1) + tf.eye(N)
        K_1 = tf.matmul(c_v, tf.transpose(c_v))
        K_2 = tf.matmul(c_w, tf.transpose(c_w))
        rst = tf.matmul(K_1, H)
        rst = tf.matmul(rst, K_2)
        rst = tf.matmul(rst, H)
        rst = tf.trace(rst)
        return rst
