import tensorflow as tf


class DOUGATE():
    def __init__(self, lambda_1, lambda_2, hidden_dims, n_sample, num_cluster, alpha):
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.n_layers = len(hidden_dims) - 1
        self.W, self.v = self.define_weights(hidden_dims)
        self.C = {}
        self.C2 = {}
        self.weight1 = tf.Variable(1.0e-4 * tf.ones(shape=(n_sample, n_sample)), name="douweight1")
        self.coef1 = self.weight1 - tf.matrix_diag(tf.diag_part(self.weight1))
        self.weight2 = tf.Variable(1.0e-4 * tf.ones(shape=(n_sample, n_sample)), name="douweight2")
        self.coef2 = self.weight2 - tf.matrix_diag(tf.diag_part(self.weight2))
        self.n_cluster = num_cluster
        self.input_batch_size = n_sample
        self.alpha = alpha

    def __call__(self, A, X, R, S, A2, X2, R2, S2):
        # Encoder1
        H = X
        for layer in range(self.n_layers):
            H = self.__encoder1(A, H, layer)
        # Final node representations
        self.H = H
        self.HC = tf.matmul(self.coef1, H)
        H = self.HC

        # Decoder1
        for layer in range(self.n_layers - 1, -1, -1):
            H = self.__decoder1(H, layer)
        X_ = H
        self.Z = self.H

        # Encoder2
        H2 = X2
        for layer in range(self.n_layers):
            H2 = self.__encoder2(A2, H2, layer)
        # Final node representations
        self.H2 = H2
        self.HC2 = tf.matmul(self.coef2, H2)
        H2 = self.HC2

        # Decoder2
        for layer in range(self.n_layers - 1, -1, -1):
            H2 = self.__decoder2(H2, layer)
        X_2 = H2
        self.Z2 = self.H2

        # The reconstruction loss of node features
        self.ft_loss1 = tf.reduce_mean((X - X_) ** 2)
        self.ft_loss2 = tf.reduce_mean((X2 - X_2) ** 2)
        self.ft_loss = self.ft_loss1 + self.ft_loss2
        # The reconstruction loss of the graph structure
        self.S_emb = tf.nn.embedding_lookup(self.H, S)  # 选取一个张量里面索引对应的元素
        self.R_emb = tf.nn.embedding_lookup(self.H, R)
        self.S_emb2 = tf.nn.embedding_lookup(self.H2, S2)
        self.R_emb2 = tf.nn.embedding_lookup(self.H2, R2)
        self.st_loss1 = -tf.log(tf.sigmoid(tf.reduce_sum(self.S_emb * self.R_emb, axis=-1)))  # reduce_sum()是按一定方式计算张量中的元素之和
        self.st_loss2 = -tf.log(tf.sigmoid(tf.reduce_sum(self.S_emb2 * self.R_emb2, axis=-1)))
        self.st_loss = tf.reduce_sum(self.st_loss1) + tf.reduce_sum(self.st_loss2)
        # The loss of self-expression and C-penalty term
        self.SE_loss1 = 0.5 * tf.reduce_mean((self.H - self.HC) ** 2)
        self.SE_loss2 = 0.5 * tf.reduce_mean((self.H2 - self.HC2) ** 2)
        self.SE_loss = self.SE_loss1 + self.SE_loss2
        self.C_Regular = tf.reduce_sum(tf.pow(tf.abs(self.coef1), 1.0)) + tf.reduce_sum(tf.pow(tf.abs(self.coef2), 1.0))
        # Total loss
        self.loss = self.ft_loss + 0.001 * self.st_loss + self.SE_loss + self.C_Regular

        return self.loss, self.ft_loss, self.st_loss, self.SE_loss, self.C_Regular

    def __encoder1(self, A, H, layer):
        H = tf.matmul(H, self.W[layer])
        self.C[layer] = self.graph_attention_layer(A, H, self.v[layer], layer)
        return tf.sparse_tensor_dense_matmul(self.C[layer], H)

    def __decoder1(self, H, layer):
        H = tf.matmul(H, self.W[layer], transpose_b=True)
        return tf.sparse_tensor_dense_matmul(self.C[layer], H)

    def __encoder2(self, A, H, layer):
        H = tf.matmul(H, self.W[layer])
        self.C2[layer] = self.graph_attention_layer(A, H, self.v[layer], layer)
        return tf.sparse_tensor_dense_matmul(self.C2[layer], H)

    def __decoder2(self, H, layer):
        H = tf.matmul(H, self.W[layer], transpose_b=True)
        return tf.sparse_tensor_dense_matmul(self.C2[layer], H)

    def define_weights(self, hidden_dims):
        W = {}
        for i in range(self.n_layers):
            W[i] = tf.get_variable("douW%s" % i, shape=(hidden_dims[i], hidden_dims[i + 1]))

        Ws_att = {}
        for i in range(self.n_layers):
            v = {}
            v[0] = tf.get_variable("douv%s_0" % i, shape=(hidden_dims[i + 1], 1))
            v[1] = tf.get_variable("douv%s_1" % i, shape=(hidden_dims[i + 1], 1))
            Ws_att[i] = v

        return W, Ws_att

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