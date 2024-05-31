import tensorflow as tf


class SINGATE():
    def __init__(self, lambda_1, lambda_2, hidden_dims, n_sample, num_cluster, alpha, id):
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.n_layers = len(hidden_dims) - 1
        if id == 1:
            self.W, self.v = self.define_weights(hidden_dims)
        else:
            self.W, self.v = self.define_weights2(hidden_dims)
        self.C = {}
        self.weight = tf.Variable(1.0e-4 * tf.ones(shape=(n_sample, n_sample)), name="sin%sweight" % id)
        # diag_part返回张量的对角线部分
        self.coef = self.weight - tf.matrix_diag(tf.diag_part(self.weight))
        self.n_cluster = num_cluster
        self.input_batch_size = n_sample
        self.alpha = alpha

    def __call__(self, A, X, R, S):
        # Encoder1
        H = X
        for layer in range(self.n_layers):
            H = self.__encoder(A, H, layer)
        # Final node representations
        self.H = H
        self.HC = tf.matmul(self.coef, H)
        H = self.HC

        # Decoder1
        for layer in range(self.n_layers - 1, -1, -1):
            H = self.__decoder(H, layer)
        X_ = H
        self.Z = self.H

        # The reconstruction loss of node features
        self.ft_loss = tf.reduce_mean((X - X_) ** 2)
        # The reconstruction loss of the graph structure
        self.S_emb = tf.nn.embedding_lookup(self.H, S)  # 选取一个张量里面索引对应的元素
        self.R_emb = tf.nn.embedding_lookup(self.H, R)
        self.st_loss = -tf.log(tf.sigmoid(tf.reduce_sum(self.S_emb * self.R_emb, axis=-1)))  # reduce_sum()是按一定方式计算张量中的元素之和
        self.st_loss = tf.reduce_sum(self.st_loss)
        # The loss of self-expression and C-penalty term
        self.SE_loss = 0.5 * tf.reduce_mean((self.H - self.HC) ** 2)
        self.C_Regular = tf.reduce_sum(tf.pow(tf.abs(self.coef), 1.0))
        # Total loss
        self.loss = self.ft_loss + self.lambda_1 * self.st_loss + self.SE_loss + self.C_Regular

        return self.loss, self.ft_loss, self.st_loss, self.SE_loss, self.C_Regular, self.Z

    def __encoder(self, A, H, layer):
        H = tf.matmul(H, self.W[layer])
        self.C[layer] = self.graph_attention_layer(A, H, self.v[layer], layer)
        return tf.sparse_tensor_dense_matmul(self.C[layer], H)

    def __decoder(self, H, layer):
        H = tf.matmul(H, self.W[layer], transpose_b=True)
        return tf.sparse_tensor_dense_matmul(self.C[layer], H)

    def define_weights(self, hidden_dims):
        W = {}
        for i in range(self.n_layers):
            W[i] = tf.get_variable("sin1W%s" % i, shape=(hidden_dims[i], hidden_dims[i + 1]))

        Ws_att = {}
        for i in range(self.n_layers):
            v = {}
            v[0] = tf.get_variable("sin1v%s_0" % i, shape=(hidden_dims[i + 1], 1))
            v[1] = tf.get_variable("sin1v%s_1" % i, shape=(hidden_dims[i + 1], 1))
            Ws_att[i] = v

        return W, Ws_att

    def define_weights2(self, hidden_dims):
        W = {}
        for i in range(self.n_layers):
            W[i] = tf.get_variable("sin2W%s" % i, shape=(hidden_dims[i], hidden_dims[i + 1]))

        Ws_att = {}
        for i in range(self.n_layers):
            v = {}
            v[0] = tf.get_variable("sin2v%s_0" % i, shape=(hidden_dims[i + 1], 1))
            v[1] = tf.get_variable("sin2v%s_1" % i, shape=(hidden_dims[i + 1], 1))
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
