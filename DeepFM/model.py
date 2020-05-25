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

    def add_embed_weights(self, vocab_name, vocab_size, embed_dim, reg):
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
                regularizer=tf.contrib.layers.l2_regularizer(reg),
                dtype=tf.float32)

            self._embed_weights[vocab_name] = embed_weight

    def get_linear_weights(self, vocab_name):
        """get linear weights"""
        return self._linear_weights[vocab_name]

    def get_embed_weights(self, vocab_name):
        """get poly weights"""
        return self._embed_weights[vocab_name]

def build_deep_layers(hidden, hidden_units, mode, reg):
  hidden = build_dnn_layers(hidden, hidden_units, mode, reg)
  logits = tf.layers.dense(inputs=hidden, units=1,
                              activation=None, use_bias=False,
                              kernel_initializer=tf.glorot_uniform_initializer(),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(reg),
                              name='PredictionLayer_{}'.format(len(hidden_units)+1))
  return logits
   
def build_dnn_layers(hidden, hidden_units, mode, reg):
  for l in range(len(hidden_units)):
    hidden = tf.layers.dense(inputs=hidden, units=hidden_units[l],
                             activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=tf.glorot_uniform_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(reg),
                             name='EmbeddingLayer_{}'.format(l+1))
    hidden = tf.layers.batch_normalization(hidden, training=True)
  return hidden 

def build_deepfm_layers(fm_part, hidden, hidden_units, mode, reg):
  deep_part = build_dnn_layers(hidden, hidden_units, mode, reg)
  all_part = tf.concat([fm_part, deep_part], 1)
  logits = tf.layers.dense(inputs=all_part, units=1,
                              activation=None, use_bias=False,
                              kernel_initializer=tf.glorot_uniform_initializer(),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(reg),                                                   
                              name='PredictionLayer_{}'.format(len(hidden_units)+1))
  return logits        


def model_fn(features, labels, mode, params):
  if params['model'] == 'deepfm':
    return build_dfm(features, labels, mode, params)

def build_dfm(features, labels, mode, params):
    """model fn"""

    # build embedding tables
    dnn_data = tf.feature_column.input_layer(features, params['dnn_columns'])
    dimension = params['embedding_size']
    cat_columns = params['cat_columns']
    val_columns = params['val_columns']
    column_to_field = params['column_to_field']
    reg = params['reg']
    hidden_units = params['hidden_units']
    weight = tf.feature_column.input_layer(features, params['weight_columns'])
    embedding_table = EmbeddingTable()
    with tf.variable_scope("dfm", reuse=tf.AUTO_REUSE, values=[features]) as scope:
        for name, col in cat_columns.items():
            field = column_to_field.get(name, name)
            embedding_table.add_linear_weights(vocab_name=name,
                                               vocab_size=col._num_buckets)
            embedding_table.add_embed_weights(vocab_name=field,
                                              vocab_size=col._num_buckets,
                                              embed_dim=dimension,
                                              reg=reg)
        for name, col in val_columns.items():
            field = column_to_field.get(name, name)
            embedding_table.add_linear_weights(vocab_name=name,
                                               vocab_size=1)
            embedding_table.add_embed_weights(vocab_name=field,
                                              vocab_size=1,
                                              embed_dim=dimension,
                                              reg=reg)

        # bias
        bias = tf.get_variable(name='bias',
                               shape=(1, ),
                               initializer=tf.constant_initializer(0.0))
    
    
        builder = _LazyBuilder(features)

        # linear part
        linear_outputs = []
        for name, col in cat_columns.items():
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
        for name, col in val_columns.items():
            dense_tensor = col._get_dense_tensor(builder)
            linear_weights = embedding_table.get_linear_weights(name)
            output = tf.multiply(dense_tensor, linear_weights)
            linear_outputs.append(output)
        # linear_outputs: (batch_szie, nonzero_feature_num)
        linear_outputs = tf.concat(linear_outputs, axis=1)
        '''
        # linear_logits: (batch_size, 1)
        linear_logits = tf.reduce_sum(linear_outputs,
                                      axis=1,
                                      keepdims=True,
                                      name='linear_logits')
        '''
        # poly part
        sum_then_square = []
        square_then_sum = []
        for name, col, in cat_columns.items():
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
        for name, col in val_columns.items():
            field = column_to_field.get(name, name)
            dense_tensor = col._get_dense_tensor(builder)
            embed_weights = embedding_table.get_embed_weights(field)
            embeddings = tf.multiply(dense_tensor, embed_weights)
            sum_then_square.append(embeddings)
            square_then_sum.append(tf.square(embeddings))
        

        print("feat num:",len(sum_then_square))
        deep_inputs = tf.concat(sum_then_square, axis = 1)
        deep_inputs = tf.concat([deep_inputs, dnn_data], axis=1)
        # sum_then_square: (batch_size, embedding)
        sum_then_square = tf.square(tf.add_n(sum_then_square))
        # square_then_sum: (batch_size, embedding)
        square_then_sum = tf.add_n(square_then_sum)
        # poly_logits: (batch_size, 1)
        poly_outputs = 0.5 * tf.subtract(sum_then_square, square_then_sum)
        fm_part = tf.concat([linear_outputs, poly_outputs], 1)
        #dnn_logits = build_deep_layers(deep_inputs, hidden_units, mode, reg)
        logits = build_deepfm_layers(fm_part, deep_inputs, hidden_units, mode, reg)
        # logits: (batch_size, 1)
        #logits = linear_logits + poly_logits + dnn_logits
        
        # predictions
        logits = tf.nn.bias_add(logits, bias)
        
    
    preds = tf.nn.sigmoid(logits)
    tf.summary.histogram("preds", preds)
    # PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
        'prob': tf.concat([1 - preds, preds], 1)
      }
      export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)  #线上预测需要>
      }
      return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)
    else:
      ctr_loss = tf.losses.log_loss(labels, preds, weights=weight, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
      optimizer = tf.train.AdamOptimizer(params['learning_rate'])
      ctr_auc = tf.metrics.auc(labels=labels, predictions=preds)
      reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
      tf.summary.scalar("loss/ctr_loss", ctr_loss)
      tf.summary.scalar("loss/reg_loss", reg_loss)
      loss = ctr_loss + reg_loss
      eval_metric_ops = {
        'auc': ctr_auc
      }
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):                                                                                             
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)
