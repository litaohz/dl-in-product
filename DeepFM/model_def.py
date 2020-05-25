"""Doc string"""

import tensorflow as tf
from tensorflow.python.feature_column.feature_column import _LazyBuilder
from tensorflow.python.ops import embedding_ops


class EmbeddingTable:
    """修改自: https://github.com/stasi009/Recommend-Estimators/blob/master/deepfm.py"""
    def __init__(self):
        self._linear_weights = {}
        self._embed_weights = {}

    def __contains__(self, item):
        return item in self._embed_weights

    def add_linear_weights(self, vocab_name, vocab_size):
        """
        :param vocab_name: 一个field拥有两个权重矩阵，一个用于线性连接，另一个用于非线性（二阶或更高阶交叉）连接
        :param vocab_size: 字典总长度
        :param embed_dim: 二阶权重矩阵shape=[vocab_size, order2dim]，映射成的embedding
                          既用于接入DNN的第一屋，也是用于FM二阶交互的隐向量
        :return: None
        """
        linear_weight = tf.get_variable(
            name='{}_linear_weight'.format(vocab_name),
            shape=[vocab_size, 1],
            initializer=tf.glorot_normal_initializer(),
            dtype=tf.float32)

        self._linear_weights[vocab_name] = linear_weight

    def add_embed_weights(self, vocab_name, vocab_size, embed_dim):
        """
        :param vocab_name: 一个field拥有两个权重矩阵，一个用于线性连接，另一个用于非线性（二阶或更高阶交叉）连接
        :param vocab_size: 字典总长度
        :param embed_dim: 二阶权重矩阵shape=[vocab_size, embed_dim]，映射成的embedding
                          既用于接入DNN的第一屋，也是用于FM二阶交互的隐向量
        :return: None
        """
        if vocab_name not in self._embed_weights:
            # 二阶（FM）特征的embedding，可共享embedding矩阵
            embed_weight = tf.get_variable(
                name='{}_embed_weight'.format(vocab_name),
                shape=[vocab_size, embed_dim],
                initializer=tf.glorot_normal_initializer(),
                dtype=tf.float32)

            self._embed_weights[vocab_name] = embed_weight

    def get_linear_weights(self, vocab_name):
        """get linear weights"""
        return self._linear_weights[vocab_name]

    def get_embed_weights(self, vocab_name):
        """get poly weights"""
        return self._embed_weights[vocab_name]


def build_fm(features, labels, mode, params):
    """model fn"""

    # build embedding tables
    dimension = params['DIMENSION']
    columns = params['FM_COLUMNS']
    column_to_field = params['COLUMN_TO_FIELD']

    embedding_table = EmbeddingTable()
    with tf.variable_scope("fm", reuse=tf.AUTO_REUSE, values=[features]) as scope:
        for name, col in columns.items():
            field = column_to_field.get(name, name)
            embedding_table.add_linear_weights(vocab_name=name,
                                               vocab_size=col._num_buckets)
            embedding_table.add_embed_weights(vocab_name=field,
                                              vocab_size=col._num_buckets,
                                              embed_dim=dimension)

        # bias
        bias = tf.get_variable(name='bias',
                               shape=(1, ),
                               initializer=tf.constant_initializer(0.0))
    
    
        builder = _LazyBuilder(features)

        # linear part
        linear_outputs = []
        for name, col in columns.items():
            # get sparse tensor of input feature from feature column
            sp_tensor = col._get_sparse_tensors(builder)
            sp_ids = sp_tensor.id_tensor

            linear_weights = embedding_table.get_linear_weights(name)

            # linear_weights: (vocab_size, 1)
            # sp_ids: (batch_size, max_tokens_per_example)
            # sp_values: (batch_size, max_tokens_per_example)
            # output: (batch_size, 1)
            output = embedding_ops.safe_embedding_lookup_sparse(
                linear_weights,
                sp_ids,
                None,
                combiner='sum',
                name='{}_linear_output'.format(name))

            linear_outputs.append(output)

        # linear_outputs: (batch_szie, nonzero_feature_num)
        linear_outputs = tf.concat(linear_outputs, axis=1)

        # linear_logits: (batch_size, 1)
        linear_logits = tf.reduce_sum(linear_outputs,
                                      axis=1,
                                      keepdims=True,
                                      name='linear_logits')
    
        # poly part
        sum_then_square = []
        square_then_sum = []
        for name, col, in columns.items():
            # get sparse tensor of input feature from feature column
            field = column_to_field.get(name, name)
            sp_tensor = col._get_sparse_tensors(builder)
            sp_ids = sp_tensor.id_tensor

            embed_weights = embedding_table.get_embed_weights(field)

            # embeddings: (batch_size, embed_dim)
            # x_i * v_i
            embeddings = embedding_ops.safe_embedding_lookup_sparse(
                embed_weights,
                sp_ids,
                None,
                combiner='sum',
                name='{}_{}_embedding'.format(field, name))
            sum_then_square.append(embeddings)
            square_then_sum.append(tf.square(embeddings))

        # sum_then_square: (batch_size, embedding)
        sum_then_square = tf.square(tf.add_n(sum_then_square))
        # square_then_sum: (batch_size, embedding)
        square_then_sum = tf.add_n(square_then_sum)

        # poly_logits: (batch_size, 1)
        poly_logits = 0.5 * tf.reduce_sum(
            tf.subtract(sum_then_square, square_then_sum), axis=1, keepdims=True)

        # logits: (batch_size, 1)
        logits = linear_logits + poly_logits

        # predictions
        logits = tf.nn.bias_add(logits, bias)
        
    
    logistic = tf.nn.sigmoid(logits, name='logistic')
    two_class_logits = tf.concat((tf.zeros_like(logits), logits),
                                 axis=-1,
                                 name='two_class_logits')
    probabilities = tf.nn.softmax(two_class_logits, name='probabilities')
    class_ids = tf.argmax(two_class_logits, axis=-1, name="class_ids")
    class_ids = tf.expand_dims(class_ids, axis=-1)

    predictions = {
        'logits': logits,
        'logistic': logistic,
        'probabilities': probabilities,
        'class_ids': class_ids
    }

    # PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

    eval_metric_ops = {
        'auc': tf.metrics.auc(labels, logistic)
    }
    # EVAL mode
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)

    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.FtrlOptimizer(learning_rate=0.01,
                                       l1_regularization_strength=0.001,
                                       l2_regularization_strength=0.001)
    global_step = tf.train.get_or_create_global_step()
    train_op = optimizer.minimize(loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)


def build_dfm(features, labels, mode, params):
    """model fn"""

    # build embedding tables
    dimension = params['DIMENSION']
    columns = params['FM_COLUMNS']
    column_to_field = params['COLUMN_TO_FIELD']

    embedding_table = EmbeddingTable()
    with tf.variable_scope("fm", reuse=tf.AUTO_REUSE, values=[features]) as scope:
        for name, col in columns.items():
            field = column_to_field.get(name, name)
            embedding_table.add_linear_weights(vocab_name=name,
                                               vocab_size=col._num_buckets)
            embedding_table.add_embed_weights(vocab_name=field,
                                              vocab_size=col._num_buckets,
                                              embed_dim=dimension)

        # bias
        bias = tf.get_variable(name='bias',
                               shape=(1, ),
                               initializer=tf.constant_initializer(0.0))
    
    
        builder = _LazyBuilder(features)

        # linear part
        linear_outputs = []
        for name, col in columns.items():
            # get sparse tensor of input feature from feature column
            sp_tensor = col._get_sparse_tensors(builder)
            sp_ids = sp_tensor.id_tensor

            linear_weights = embedding_table.get_linear_weights(name)

            # linear_weights: (vocab_size, 1)
            # sp_ids: (batch_size, max_tokens_per_example)
            # sp_values: (batch_size, max_tokens_per_example)
            # output: (batch_size, 1)
            output = embedding_ops.safe_embedding_lookup_sparse(
                linear_weights,
                sp_ids,
                None,
                combiner='sum',
                name='{}_linear_output'.format(name))

            linear_outputs.append(output)

        # linear_outputs: (batch_szie, nonzero_feature_num)
        linear_outputs = tf.concat(linear_outputs, axis=1)

        # linear_logits: (batch_size, 1)
        linear_logits = tf.reduce_sum(linear_outputs,
                                      axis=1,
                                      keepdims=True,
                                      name='linear_logits')
    
        # poly part
        sum_then_square = []
        square_then_sum = []
        for name, col, in columns.items():
            # get sparse tensor of input feature from feature column
            field = column_to_field.get(name, name)
            sp_tensor = col._get_sparse_tensors(builder)
            sp_ids = sp_tensor.id_tensor

            embed_weights = embedding_table.get_embed_weights(field)

            # embeddings: (batch_size, embed_dim)
            # x_i * v_i
            embeddings = embedding_ops.safe_embedding_lookup_sparse(
                embed_weights,
                sp_ids,
                None,
                combiner='sum',
                name='{}_{}_embedding'.format(field, name))
            sum_then_square.append(embeddings)
            square_then_sum.append(tf.square(embeddings))

        # sum_then_square: (batch_size, embedding)
        sum_then_square = tf.square(tf.add_n(sum_then_square))
        # square_then_sum: (batch_size, embedding)
        square_then_sum = tf.add_n(square_then_sum)

        # poly_logits: (batch_size, 1)
        poly_logits = 0.5 * tf.reduce_sum(
            tf.subtract(sum_then_square, square_then_sum), axis=1, keepdims=True)

        # logits: (batch_size, 1)
        logits = linear_logits + poly_logits

        # predictions
        logits = tf.nn.bias_add(logits, bias)
        
    
    logistic = tf.nn.sigmoid(logits, name='logistic')
    two_class_logits = tf.concat((tf.zeros_like(logits), logits),
                                 axis=-1,
                                 name='two_class_logits')
    probabilities = tf.nn.softmax(two_class_logits, name='probabilities')
    class_ids = tf.argmax(two_class_logits, axis=-1, name="class_ids")
    class_ids = tf.expand_dims(class_ids, axis=-1)

    predictions = {
        'logits': logits,
        'logistic': logistic,
        'probabilities': probabilities,
        'class_ids': class_ids
    }

    # PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

    eval_metric_ops = {
        'auc': tf.metrics.auc(labels, logistic)
    }
    # EVAL mode
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)

    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.FtrlOptimizer(learning_rate=0.01,
                                       l1_regularization_strength=0.001,
                                       l2_regularization_strength=0.001)
    global_step = tf.train.get_or_create_global_step()
    train_op = optimizer.minimize(loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)
