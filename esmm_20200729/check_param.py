#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import sys
import os

export_dir = sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

with tf.Session() as sess:
  meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
  signature = meta_graph_def.signature_def
  signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
  
  input_name = signature[signature_key].inputs['inputs'].name
  output_name = signature[signature_key].outputs['prob'].name
  #print(input_name, output_name)
  #print(sess.graph.get_tensor_by_name(input_name))
  #ctcvr_preds = sess.graph.get_tensor_by_name("mul_2:0")
  #ctr_preds = sess.graph.get_tensor_by_name("Sigmoid:0")
  #input_x = sess.graph.get_tensor_by_name(input_name)
  #predict_y = sess.graph.get_tensor_by_name(output_name)
  
  #print(tf.concat([ctr_preds,ctcvr_preds], axis=1))
  for node in sess.graph_def.node:
    print(node.name)
  #print(signature[signature_key].inputs, signature[signature_key].outputs)
  #print(input_x, predict_y)
