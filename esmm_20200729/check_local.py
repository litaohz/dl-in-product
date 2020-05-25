#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import sys
import os
import argparse
import configparser
sys.path.append("./tftools")
from Trainer.InputFn import input_fn

from FCGen import FCGen

def configure(parser):
  parser.add_argument(
    '--conf', type=str, help='param configuration file is requried'
  )


old_export_dir = 'esmm_export/1591364162'
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

parser = argparse.ArgumentParser()
configure(parser)
FLAGS, _ = parser.parse_known_args()

config = configparser.ConfigParser()
config.read(FLAGS.conf)

seed = int(config['train'].get('seed', 19910825))
tf.set_random_seed(seed)

train_dir_list = config['input']['dev'].split(',')
train_files = [os.path.join(train_dir, f) for train_dir in train_dir_list
                                          for f in os.listdir(train_dir) if f != "_SUCCESS"]

feature_config = configparser.ConfigParser()
feature_config.read(config['input']['spec'])
columns, spec = FCGen.GetFeatureSpec(feature_config)
train_inputs = input_fn(train_files, spec, shuffle=False, batch_size=8, mt=True)
features = train_inputs['features']
#batch_input = tf.feature_column.input_layer(features, list(columns['dnn'].values()))
labels = train_inputs['labels']
example = train_inputs['example']
with tf.Session() as sess:
  meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], old_export_dir)
  signature = meta_graph_def.signature_def
  signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
  sess.run([tf.global_variables_initializer(), train_inputs['iterator_init_op']])
  batch,label,feat = sess.run([example,labels,features])
  #print(len(feat))
  #print(feat.keys())
  #for f in feat:
  #  print(type(f))
  
  input_name = signature[signature_key].inputs['inputs'].name
  output_name = signature[signature_key].outputs['prob'].name
  input_x = sess.graph.get_tensor_by_name(input_name)
  ctcvr_preds = sess.graph.get_tensor_by_name("mul_2:0")
  ctr_preds = sess.graph.get_tensor_by_name("Sigmoid:0")
  predict_y = tf.concat([ctr_preds,ctcvr_preds], axis=1)
  print(sess.run(predict_y, feed_dict={input_x:batch}))
  print(label)
  
  #print(signature[signature_key].inputs, signature[signature_key].outputs)
  #print(input_x, predict_y)
