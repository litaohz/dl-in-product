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
                              name='{}PredictionLayer_{}'.format(task, l+1))
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
  logits = tf.layers.dense(inputs=all_part, units=1,
                              activation=None, use_bias=False,
                              kernel_initializer=tf.glorot_uniform_initializer(),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(reg),
                              name='{}PredictionLayer_{}'.format(task, len(hidden_units)+1))
  return logits
  

def dssm_model_fn(features, labels, mode, params):
  batch_user = tf.feature_column.input_layer(features, params['user_columns'])
  batch_item = tf.feature_column.input_layer(features, params['item_columns'])
  print("batch_user data shape, ", batch_user.get_shape().as_list())
  print("batch_item data shape, ", batch_item.get_shape().as_list())
  hidden_units = params['hidden_units']
  ctr_reg = params['ctr_reg']
  user_last_layer = build_dnn_layers(batch_user, hidden_units, mode, 'User', ctr_reg)
  item_last_layer = build_dnn_layers(batch_item, hidden_units, mode, 'Item', ctr_reg)
  ctr_reg = params['ctr_reg']
  cvr_reg = params['cvr_reg']
  user_layer_norm = tf.nn.l2_normalize(tf.nn.relu(user_last_layer), axis=1, epsilon=1e-8, name="user_layer_norm")
  item_layer_norm = tf.nn.l2_normalize(tf.nn.relu(item_last_layer), axis=1, epsilon=1e-8, name="item_layer_norm")

  with tf.name_scope('Cosine_Similarity'):
      outputs = tf.reduce_sum(tf.multiply(user_layer_norm, item_layer_norm), axis=1, keepdims=True, name='cos')
      # prob = tf.nn.softmax(get_cosine_score(user_last_layer, item_last_layer))
      # prob = tf.clip_by_value(prob, 1e-8, 1.0)
#       outputs = tf.layers.dense(output, units=1, kernel_constraint=tf.keras.constraints.NonNeg(), name="fc_loss")

      tf.summary.histogram('user_layer_norm', user_layer_norm)
      tf.summary.histogram('item_layer_norm', item_layer_norm)
      # tf.summary.scalar('item_layer_norm', tf.squeeze(item_layer_norm))
      # tf.summary.scalar('prob', prob)
      # tf.summary.scalar('output', output)
  prob = tf.nn.sigmoid(outputs)
  print("prob:", prob)

  if mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
          "user_last_layer": user_layer_norm,
          "item_last_layer": item_layer_norm,
          "prob": prob,
      }
      export_outputs = {
        'prediction': tf.estimator.export.PredictOutput(predictions)
      }
      return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

  with tf.name_scope('Loss'):
      # import pdb;pdb.set_trace()
      # cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels,dtype=tf.float32), logits=prob)
      # loss = tf.reduce_sum(cross_entropy)
      # cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels,dtype=tf.float32), logits=prob)
      # loss = tf.reduce_sum(cross_entropy)
      # loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, dtype=tf.float32), logits=output, name="loss"))
          # loss = tf.reduce_mean(losse))
      losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, dtype=tf.float32), logits=outputs, name="scl_loss")
      loss = tf.reduce_mean(losses)
  
  auc = tf.metrics.auc(labels, prob)
  metrics = {'auc': auc}
  tf.summary.scalar('auc_0', auc[0])
  tf.summary.scalar('auc_1', auc[1])
#   tf.summary.histogram('output_0', output)
  tf.summary.histogram('output_1', outputs)
  


  if mode == tf.estimator.ModeKeys.EVAL:
      return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

  # Create training op.
  assert mode == tf.estimator.ModeKeys.TRAIN
  optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
  return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)

