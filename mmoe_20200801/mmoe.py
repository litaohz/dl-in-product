#coding:utf-8
"""train the model"""

import argparse
import configparser
import logging
import reprlib
import sys
import os

import tensorflow as tf
from tensorflow.python.feature_column.feature_column import _LazyBuilder
from tensorflow.python.ops import embedding_ops
sys.path.append("../ESMM/tftools/")

from FCGen import FCGen


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
    :param embed_dims: 二阶权重矩阵shape=[vocab_size, embed_dim]，映射成的embedding
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

def build_deep_layers(hidden, hidden_units, mode, task, reg):
  hidden = build_dnn_layers(hidden, hidden_units, mode, task, reg)
  logits = tf.layers.dense(inputs=hidden, units=1,
                              activation=None, use_bias=False,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(reg),
                              name='{}PredictionLayer_{}'.format(task, len(hidden_units)+1))
  return logits

def build_dnn_layers(hidden, hidden_units, mode, task, reg):
  for l in range(len(hidden_units)):
    hidden = tf.layers.dense(inputs=hidden, units=hidden_units[l],
                             activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=tf.glorot_uniform_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(reg),
                             name='{}EmbeddingLayer_{}'.format(task, l+1))
    hidden = tf.layers.batch_normalization(hidden, training=True)
  return hidden

def build_fm_layers(hidden, hidden_units, embedding_size):
  num_feat = int(hidden.get_shape().as_list()[-1] / embedding_size)
  linear_weight = tf.get_variable(name='linear_weight',
    shape=[num_feat],
    initializer=tf.truncated_normal_initializer(0.0, 0.1))
  embedding = tf.reshape(hidden, [-1, num_feat, embedding_size])
  first_order_input = tf.reduce_mean(embedding, 2)
  first_order_output = tf.multiply(first_order_input, linear_weight)
  second_order_sum_square = tf.square(tf.reduce_sum(embedding, 1))
  second_order_square_sum = tf.reduce_sum(tf.square(embedding), 1)
  second_order_output = 0.5 * tf.subtract(second_order_sum_square, second_order_square_sum)
  fm_part = tf.concat([first_order_output, second_order_output], 1)
  return fm_part

def build_deepfm_layers(fm_part, hidden, hidden_units, mode, task, reg):
  deep_part = build_dnn_layers(hidden, hidden_units, mode, task, reg)
  all_part = tf.concat([fm_part, deep_part], 1)
  return all_part

def build_mmoe(features, params, scope, task_num=2):
  with tf.variable_scope(scope):
    inputs = build_input(features, params)
    expert_num = params['expert_num']
    expert_unit = params['expert_unit']
    reg = params['reg']
    experts_weight = tf.get_variable(name="experts_weight", dtype=tf.float32,
                                     shape=(inputs.get_shape()[1], expert_unit, expert_num),
                                     initializer=tf.glorot_normal_initializer(),
                                     regularizer=tf.contrib.layers.l2_regularizer(reg))
    experts_bias = tf.get_variable(name="experts_bias", dtype=tf.float32,
                                   shape=(expert_unit, expert_num),
                                   initializer=tf.glorot_normal_initializer(),
                                   regularizer=tf.contrib.layers.l2_regularizer(reg))
    experts_output = tf.nn.relu(tf.add(tf.tensordot(inputs, experts_weight, axes=1), experts_bias))

    gates_weight = [tf.get_variable(name="gate_weight_task_{}".format(i), dtype=tf.float32,
                                    shape=(inputs.get_shape()[1], expert_num),
                                    initializer=tf.glorot_normal_initializer())
                    for i in range(task_num)]
    gates_bias = [tf.get_variable(name="gate_bias_task_{}".format(i), dtype=tf.float32,
                                  shape=(expert_num,), initializer=tf.glorot_normal_initializer())
                  for i in range(task_num)]
    gates_output = [tf.nn.softmax(tf.add(tf.matmul(inputs, gates_weight[i]), gates_bias[i]))
                    for i in range(task_num)]
    multi_inputs = [tf.reshape(
      tf.reduce_sum(
        tf.multiply(experts_output, tf.expand_dims(gates_output[i], axis=1)),
        axis=2),
      [-1, experts_output.get_shape()[1]])
      for i in range(task_num)]
    return multi_inputs, experts_weight

def build_input(features, params):
  cat_columns = params['cat_columns']
  val_columns = params['val_columns']
  column_to_field = params['column_to_field']
  dnn_columns = params['dnn_columns']
  reg = params['reg']
  embed_dim = params['embed_dim']

  dnn_part = tf.feature_column.input_layer(features, dnn_columns)
  embedding_table = EmbeddingTable()
  with tf.variable_scope("fm", reuse=tf.AUTO_REUSE, values=[features]) as scope:
    with tf.device('/cpu:0'):
      for name, col in cat_columns.items():
        field = column_to_field.get(name, name)
        embedding_table.add_linear_weights(vocab_name=name,
                                           vocab_size=col._num_buckets)
        embedding_table.add_embed_weights(vocab_name=field,
                                          vocab_size=col._num_buckets,
                                          embed_dim=embed_dim,
                                          reg=reg)
      for name, col in val_columns.items():
        field = column_to_field.get(name, name)
        embedding_table.add_linear_weights(vocab_name=name,
                                           vocab_size=1)
        embedding_table.add_embed_weights(vocab_name=field,
                                          vocab_size=1,
                                          embed_dim=embed_dim,
                                          reg=reg)

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
        linear_output = embedding_ops.safe_embedding_lookup_sparse(
          linear_weights,
          sp_ids,
          None,
          combiner='sum',
          name='{}_linear_output'.format(name))

        linear_outputs.append(linear_output)
      for name, col in val_columns.items():
        dense_tensor = col._get_dense_tensor(builder)
        linear_weights = embedding_table.get_linear_weights(name)
        linear_output = tf.multiply(dense_tensor, linear_weights)
        linear_outputs.append(linear_output)
      # linear_outputs: (batch_szie, nonzero_feature_num)
      linear_outputs = tf.concat(linear_outputs, axis=1)
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

    # sum_then_square: (batch_size, embedding)
    sum_then_square = tf.square(tf.add_n(sum_then_square))
    # square_then_sum: (batch_size, embedding)
    square_then_sum = tf.add_n(square_then_sum)
    poly_outputs = 0.5 * tf.subtract(sum_then_square, square_then_sum)
    new_inputs = tf.concat([linear_outputs, poly_outputs, dnn_part], 1)

    return new_inputs

def grad_norm(loss_list, weights_shared):
  alpha = 0.12
  task_num = len(loss_list)
  w_list = [tf.Variable(1.0, name="w_".format(i)) for i in range(task_num)]
  weight_loss = [tf.multiply(loss_list[i], w_list[i]) for i in range(task_num)]
  init_list = [tf.Variable(-1.0, trainable=False) for i in range(task_num)]
  def assign_init(init, loss):
    with tf.control_dependencies([tf.assign(init, loss)]):
      return tf.identity(init)
  orig_list = [tf.cond(
                    tf.equal(init_list[i], -1.0),
                    lambda: assign_init(init_list[i], loss_list[i]),
                    lambda: tf.identity(init_list[i]))
                for i in range(task_num)]
  G_norm_list = [tf.norm(tf.gradients(weight_loss[i], weights_shared), ord=2) 
                for i in range(task_num)]
  G_avg = tf.div(tf.add_n(G_norm_list), task_num)
  l_hat_list = [tf.div(loss_list[i], orig_list[i]) for i in range(task_num)]
  l_hat_avg = tf.div(tf.add_n(l_hat_list), task_num)
  inv_rate_list = [tf.div(l_hat_list[i], l_hat_avg) for i in range(task_num)]
  a = tf.constant(alpha)
  C_list = [tf.multiply(G_avg, tf.pow(inv_rate_list[i], a)) for i in range(task_num)]
  loss_gradnorm = tf.add_n([tf.reduce_sum(tf.abs(tf.subtract(G_norm_list[i], C_list[i])))
                            for i in range(task_num)])
  with tf.control_dependencies([loss_gradnorm]):
    coef = tf.div(float(task_num), tf.add_n(w_list))
    update_list = [w_list[i].assign(tf.multiply(w_list[i], coef))
                    for i in range(task_num)]
  for i in range(task_num):
    tf.summary.scalar("gradnorm/w_{}".format(i), tf.squeeze(w_list[i]))
  return weight_loss, update_list, w_list, loss_gradnorm

def model_fn(features, labels, mode, params):
  batch_weight = tf.feature_column.input_layer(features, params['weight_columns'])
  mmoe_scope = 'mmoe'
  multi_inputs, weights_shared = build_mmoe(features, params, mmoe_scope, task_num=3)
  hidden_units = params['hidden_units']
  ctr_reg = params['ctr_reg']
  ctcvr_reg = params['ctcvr_reg']
  cvr_reg = params['cvr_reg']
  #if params['model'] == 'dnn':
  dnn_scope = 'dnn'
  mask = tf.map_fn(lambda x:tf.cond(tf.equal(x ,1), lambda: True, lambda: False), tf.squeeze(labels['ctr']), dtype=tf.bool)
  cvr_inputs = tf.boolean_mask(multi_inputs[2], mask)
  with tf.variable_scope(dnn_scope):
    ctr_logits = build_deep_layers(multi_inputs[0], hidden_units, mode, 'CTR', ctr_reg)
    ctcvr_logits = build_deep_layers(multi_inputs[1], hidden_units, mode, 'CTCVR', ctcvr_reg)
    cvr_logits = build_deep_layers(cvr_inputs, hidden_units[1:], mode, 'CVR', cvr_reg)
  ctr_preds = tf.nn.sigmoid(ctr_logits)
  ctcvr_preds = tf.nn.sigmoid(ctcvr_logits)
  cvr_preds = tf.nn.sigmoid(cvr_logits)
  #ctcvr_preds = tf.stop_gradient(ctr_preds) * cvr_preds
  #ctcvr_preds = ctr_preds * cvr_preds
  tf.summary.histogram("mmoe/ctr_preds", ctr_preds) 
  tf.summary.histogram("mmoe/ctcvr_preds", ctcvr_preds)
  tf.summary.histogram("mmoe/cvr_preds", cvr_preds)
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
      'prob': tf.concat([cvr_preds, ctr_preds], 1)
    }
    export_outputs = {
      tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)  #线上预测需要的
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

  else:
    ctr_labels = labels['ctr']
    ctcvr_labels = labels['ctcvr']
    cvr_labels = tf.boolean_mask(ctcvr_labels, mask)
    optimizer = tf.train.AdamOptimizer(params['learning_rate'])
    loss_optimizer = tf.train.AdamOptimizer(params['learning_rate'])
    ctr_auc = tf.metrics.auc(labels=ctr_labels, predictions=ctr_preds, weights=batch_weight)
    ctcvr_auc = tf.metrics.auc(labels=ctcvr_labels, predictions=ctcvr_preds)
    cvr_auc = tf.metrics.auc(labels=cvr_labels, predictions=cvr_preds)
    ctr_loss = tf.losses.log_loss(ctr_labels, ctr_preds, weights=batch_weight, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
    ctcvr_loss = tf.losses.log_loss(ctcvr_labels, ctcvr_preds, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
    cvr_loss = tf.losses.log_loss(cvr_labels, cvr_preds, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
    reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    #loss = ctr_loss + ctcvr_loss + reg_loss
    weight_loss, update_list, w_list, loss_gradnorm = grad_norm([ctr_loss, ctcvr_loss, cvr_loss], weights_shared)
    loss = tf.add_n(weight_loss + [reg_loss])
    tf.summary.scalar("loss/ctr_loss", ctr_loss)
    tf.summary.scalar("loss/ctcvr_loss", ctcvr_loss)
    tf.summary.scalar("loss/cvr_loss", cvr_loss)
    tf.summary.scalar("loss/loss", loss)
    metrics = {'ctr_auc': ctr_auc, 'ctcvr_auc': ctcvr_auc, 'cvr_auc': cvr_auc}
    print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dnn_scope))
    print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=mmoe_scope))
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dnn_scope) + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=mmoe_scope)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_ops = []
    train_ops.append(loss_optimizer.minimize(loss_gradnorm, var_list=w_list))
    train_ops.append(update_list)
    with tf.control_dependencies(update_ops):
      train_ops.append(optimizer.minimize(loss, global_step=tf.train.get_global_step(), var_list=var_list))
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=tf.group(*train_ops), eval_metric_ops=metrics)
