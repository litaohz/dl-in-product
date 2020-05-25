#coding:utf-8
"""train the model"""

import argparse
import configparser
import logging
import reprlib
import sys
import os

import tensorflow as tf


sys.path.append("tftools/")

from FCGen import FCGen

def build_deep_layers(hidden, hidden_units, mode, task, reg):
  hidden = build_dnn_layers(hidden, hidden_units, mode, task, reg)
  logits = tf.layers.dense(inputs=hidden, units=1,
                              activation=None, use_bias=False,
                              kernel_initializer=tf.glorot_uniform_initializer(),
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
  first_order_input = tf.reduce_sum(embedding, 2)
  first_order_output = tf.multiply(first_order_input, linear_weight)
  second_order_sum_square = tf.square(tf.reduce_sum(embedding, 1))
  second_order_square_sum = tf.reduce_sum(tf.square(embedding), 1)
  second_order_output = 0.5 * tf.subtract(second_order_sum_square, second_order_square_sum)
  fm_part = tf.concat([first_order_output, second_order_output], 1)
  return fm_part

def build_deepfm_layers(fm_part, hidden, hidden_units, mode, task, reg):
  deep_part = build_dnn_layers(hidden, hidden_units, mode, task, reg)
  all_part = tf.concat([fm_part, deep_part], 1)
  logits = tf.layers.dense(inputs=all_part, units=1,
                              activation=None, use_bias=False,
                              kernel_initializer=tf.glorot_uniform_initializer(),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(reg),
                              name='{}PredictionLayer_{}'.format(task, len(hidden_units)+1))
  return logits
  

def esmm_model_fn(features, labels, mode, params):
  batch_data = tf.feature_column.input_layer(features, params['feature_columns'])
  batch_weight = tf.feature_column.input_layer(features, params['weight_columns'])
  print("data shape, ", batch_data.get_shape().as_list())
  hidden_units = params['hidden_units']
  ctr_reg = params['ctr_reg']
  cvr_reg = params['cvr_reg']
  if params['model'] == 'dnn':
    ctr_logits = build_deep_layers(batch_data, hidden_units, mode, 'CTR', ctr_reg)
    cvr_logits = build_deep_layers(batch_data, hidden_units, mode, 'CVR', cvr_reg)
  else:
    fm_part = build_fm_layers(batch_data, hidden_units, params['embedding_size'])
    ctr_logits = build_deepfm_layers(fm_part, batch_data, hidden_units, mode, 'CTR', ctr_reg)
    cvr_logits = build_deepfm_layers(fm_part, batch_data, hidden_units, mode, 'CVR', cvr_reg)
  ctr_preds = tf.nn.sigmoid(ctr_logits)
  cvr_preds = tf.nn.sigmoid(cvr_logits)
  #ctcvr_preds = tf.stop_gradient(ctr_preds) * cvr_preds
  ctcvr_preds = ctr_preds * cvr_preds
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
    #for variable_name in tf.trainable_variables():
    #  print(variable_name)
    ctr_labels = labels['ctr']
    ctcvr_labels = labels['ctcvr']
    optimizer = tf.train.AdamOptimizer(params['learning_rate'])
    ctr_auc = tf.metrics.auc(labels=ctr_labels, predictions=ctr_preds)
    ctcvr_auc = tf.metrics.auc(labels=ctcvr_labels, predictions=ctcvr_preds)
    ctr_precision, ctr_precision_update_op = tf.metrics.precision_at_thresholds(labels=ctr_labels, predictions=ctr_preds, thresholds=[0.5])
    ctr_recall, ctr_recall_update_op = tf.metrics.recall_at_thresholds(labels=ctr_labels, predictions=ctr_preds, thresholds=[0.5])
    ctr_loss = tf.losses.log_loss(ctr_labels, ctr_preds, weights=batch_weight, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
    ctcvr_loss = tf.losses.log_loss(ctcvr_labels, ctcvr_preds, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
    reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    tf.summary.scalar("esmm/ctr_loss", ctr_loss)
    tf.summary.scalar("esmm/ctcvr_loss", ctcvr_loss)
    loss = ctr_loss + params['ctcvr_loss_weight'] * ctcvr_loss + reg_loss
    metrics = {'ctr_auc': ctr_auc, 'ctcvr_auc': ctcvr_auc, 'ctr_precision':(ctr_precision[0], ctr_precision_update_op), 'ctr_recall':(ctr_recall[0],ctr_recall_update_op)}
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)
