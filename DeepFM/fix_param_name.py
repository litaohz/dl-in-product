#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import sys
import os

old_export_dir = sys.argv[1]
new_export_dir = sys.argv[2]
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

builder = tf.saved_model.builder.SavedModelBuilder(new_export_dir)
with tf.Session() as sess:
  meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], old_export_dir)
  signature = meta_graph_def.signature_def
  signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
  
  input_name = signature[signature_key].inputs['inputs'].name
  output_name = signature[signature_key].outputs['prob'].name
  input_x = sess.graph.get_tensor_by_name(input_name)
  prob = sess.graph.get_tensor_by_name("concat:0")
  model_input = tf.saved_model.utils.build_tensor_info(input_x)
  model_output = tf.saved_model.utils.build_tensor_info(prob)
  prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'inputs': model_input},
                outputs={'prob': model_output},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

  builder.add_meta_graph_and_variables(sess=sess, tags=[tf.saved_model.tag_constants.SERVING], 
                signature_def_map={signature_key:prediction_signature})

  builder.save()
  
  #print(signature[signature_key].inputs, signature[signature_key].outputs)
  #print(input_x, predict_y)
