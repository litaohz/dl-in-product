#coding:utf-8
"""train the model"""

import argparse
import configparser
import logging
import reprlib
import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.feature_column.feature_column import _LazyBuilder
from tensorflow.python.ops import embedding_ops
from tensorflow.python.estimator.canned import dnn, linear, optimizers
from tensorflow.python.training import distribute as distribute_lib
import six

sys.path.append("tftools/")

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

  def get_linear_weights(self, vocab_name=None):
    """get linear weights"""
    if vocab_name is not None:
      return self._linear_weights[vocab_name]
    else:
      return self._linear_weights

  def get_embed_weights(self, vocab_name=None):
    """get poly weights"""
    if vocab_name is not None:
      return self._embed_weights[vocab_name]
    else:
      return self._embed_weights

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
    hidden = tf.layers.batch_normalization(hidden, momentum=0.999, training=mode == tf.estimator.ModeKeys.TRAIN)
  return hidden  

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
    
    return new_inputs,embedding_table

total = 34848
def lbtw(loss_list):
  # Loss-Balanced Task Weighting to Reduce Negative Transfer in Multi-Task Learning
  task_num = len(loss_list)
  init_list = [tf.Variable(-1.0, trainable=False) for i in range(task_num)]
  total_tensor = tf.get_variable("total",initializer=tf.constant(total))
  step = tf.mod(tf.train.get_global_step(), total_tensor)
  def assign_init(init, loss):
    with tf.control_dependencies([tf.assign(init, loss)]):
      return tf.identity(init)
  alpha = 0.5
  orig_list = [tf.cond(tf.equal(step, 0), 
                        lambda: assign_init(init_list[i], loss_list[i]),
                        lambda: tf.identity(init_list[i]))
                for i in range(task_num)]
  l_hat_list = [tf.div(loss_list[i], orig_list[i]) for i in range(task_num)]
  l_hat_avg = tf.div(tf.add_n(l_hat_list), task_num)
  inv_rate_list = [tf.div(l_hat_list[i], l_hat_avg) for i in range(task_num)]
  a = tf.constant(alpha)
  w_list = [tf.pow(inv_rate_list[i], a) for i in range(task_num)]
  weight_loss = [tf.multiply(loss_list[i], w_list[i]) for i in range(task_num)]
  return weight_loss
  
def get_weight_loss(loss_list, dynamic, weights_shared):
  if dynamic:
    return grad_norm(loss_list, weights_shared)
  return loss_list, None, None, None

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
  G_norm_list = []
  for i in range(task_num):
    g_norm = []
    for _, weights in weights_shared.items():
      grads = tf.gradients(weight_loss[i], weights)
      #g_norm.append(tf.norm(tf.concat(grads,axis=0), ord=2))
      g_norm.append(tf.stack([tf.norm(tf.add(grad,1e-8), ord=2) for grad in grads]))
      #g_norm.append(tf.concat([tf.norm(tf.gradients(weight_loss[i], weight)[0],ord=2) for weight in weights], axis=0))
    G_norm_list.append(tf.stack(g_norm))
  #G_norm_list = [tf.norm(tf.gradients(weight_loss[i], weights_shared), ord=2)
  #              for i in range(task_num)]
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


def esmm_model_fn(features, labels, mode, params):
  batch_weight = tf.feature_column.input_layer(features, params['weight_columns'])
  inputs, embedding_table = build_input(features, params)
  hidden_units = params['hidden_units']
  linear_parent_scope = 'linear'
  dnn_parent_scope = 'dnn'
  is_dynamic = params['dynamic']
  print("is_dynamic:", is_dynamic)
  reg = 1e-4
  if params['model'] == 'linear':
    with tf.variable_scope(linear_parent_scope, values=tuple(six.itervalues(features)), reuse=tf.AUTO_REUSE):
      with tf.variable_scope('linear_ctr'):
        ctr_logit_fn = linear._linear_logit_fn_builder(1, params['linear_columns'])
        ctr_logits = ctr_logit_fn(features=features)
      with tf.variable_scope('linear_cvr'):
        cvr_logit_fn = linear._linear_logit_fn_builder(1, params['linear_columns'])
        cvr_logits = cvr_logit_fn(features=features)
  if params['model'] == 'dnn':
    with tf.variable_scope(dnn_parent_scope):
      with tf.variable_scope('dnn_ctr'):
        ctr_logits = build_deep_layers(inputs, hidden_units, mode, params['ctr_reg'])
        #ctr_logit_fn = dnn._dnn_logit_fn_builder(1, hidden_units, params['dnn_columns'], tf.nn.relu, None, None, True)
        #ctr_logits = ctr_logit_fn(features=features, mode=mode)
      with tf.variable_scope('dnn_cvr'):
        cvr_logits = build_deep_layers(inputs, hidden_units, mode, params['cvr_reg'])
        #cvr_logit_fn = dnn._dnn_logit_fn_builder(1, hidden_units, params['dnn_columns'], tf.nn.relu, None, None, True)
        #cvr_logits = cvr_logit_fn(features=features, mode=mode)
  ctr_preds = tf.nn.sigmoid(ctr_logits)
  cvr_preds = tf.nn.sigmoid(cvr_logits)
  ctcvr_preds = tf.stop_gradient(ctr_preds) * cvr_preds
  #ctcvr_preds = ctr_preds * cvr_preds
  tf.summary.histogram("esmm/ctr_preds", ctr_preds) 
  tf.summary.histogram("esmm/ctcvr_preds", ctcvr_preds)
  if mode == tf.estimator.ModeKeys.PREDICT:
    #redundant_items = ctr_preds
    predictions = {
      'prob': tf.concat([ctcvr_preds, ctr_preds], 1)
    }
    export_outputs = {
      tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)  #线上预测需要的
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

  else:
    #for variable in tf.trainable_variables('fm'):
    #  print(variable_name)
    #print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fm'))
    #print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dnn_parent_scope))
    #shared_weights = tf.trainable_variables(dnn_parent_scope + '/SharedLayer/kernel')[0]
    linear_weights = list(embedding_table.get_linear_weights().values())
    embed_weights = list(embedding_table.get_embed_weights().values())
    shared_weights = {'linear':linear_weights, 'embed':embed_weights}
    ctr_labels = labels['ctr']
    ctcvr_labels = labels['ctcvr']
    linear_optimizer = tf.train.FtrlOptimizer(0.01, l1_regularization_strength=0.01, l2_regularization_strength=0.001)
    dnn_optimizer = optimizers.get_optimizer_instance('Adam', params['learning_rate'])
    loss_optimizer = optimizers.get_optimizer_instance('Adam', 0.001)
    ctr_auc = tf.metrics.auc(labels=ctr_labels, predictions=ctr_preds)
    ctcvr_auc = tf.metrics.auc(labels=ctcvr_labels, predictions=ctcvr_preds)
    ctr_precision, ctr_precision_update_op = tf.metrics.precision(labels=ctr_labels, predictions=ctr_preds)
    ctr_recall, ctr_recall_update_op = tf.metrics.recall(labels=ctr_labels, predictions=ctr_preds)
    ctr_loss = tf.losses.log_loss(ctr_labels, ctr_preds, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE, weights=batch_weight)
    ctcvr_loss = tf.losses.log_loss(ctcvr_labels, ctcvr_preds, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
    reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    weight_loss, update_list, w_list, loss_gradnorm = get_weight_loss([ctr_loss, ctcvr_loss], is_dynamic, shared_weights)
    print("get_weight_loss:", weight_loss, update_list)
    loss = tf.add_n(weight_loss + [reg_loss])
    tf.summary.scalar("esmm/ctr_loss", tf.reduce_sum(ctr_loss))
    tf.summary.scalar("esmm/ctcvr_loss", tf.reduce_sum(ctcvr_loss))
    tf.summary.scalar("esmm/loss", tf.reduce_sum(loss))
    def _train_op_fn(loss):
      train_ops = []
      global_step = tf.train.get_global_step()
      if params['model'] in ('dnn'):
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fm') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dnn_parent_scope)
        train_ops.append(
          dnn_optimizer.minimize(
            loss,
            var_list=var_list))
      if params['model'] in ('linear'):
        train_ops.append(
          linear_optimizer.minimize(
              loss,
              var_list=tf.get_collection(
                  tf.GraphKeys.TRAINABLE_VARIABLES,
                  scope=linear_parent_scope)))
      if w_list and update_list and loss_gradnorm:
        train_ops.append(
            loss_optimizer.minimize(
                loss_gradnorm,
                var_list=w_list))
        train_ops.append(update_list)
      train_op = tf.group(*train_ops)
      with tf.control_dependencies([train_op]):
        return distribute_lib.increment_var(global_step)
    metrics = {'ctr_auc': ctr_auc, 'ctcvr_auc': ctcvr_auc, 'ctr_precision':(ctr_precision, ctr_precision_update_op), 'ctr_recall':(ctr_recall, ctr_recall_update_op)}
    train_op = _train_op_fn(loss)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
      train_op = tf.group(train_op, *update_ops)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)
