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
#sys.path.append("/data4/timmyqiu/qmkg/TensorFlowRec/tftools")
sys.path.append("./tftools")
from Trainer.InputFn import input_fn

from FCGen import FCGen

def configure(parser):
  parser.add_argument(
    '--conf', type=str, help='param configuration file is requried'
  )

 

def get_one(key, train_inputs):
  with tf.Session() as sess:
    sess.run(train_inputs['iterator_init_op'])
    next_element = train_inputs['features']

    sample = sess.run([
        next_element[key],
    ])
    print(sample)
    
        
def get_spec(config):
  """Entry for trainig

  Args:
    config: (configparser) All the hyperparameters for training
  """
  train_dir_list = config['input']['train'].split(',')
  train_files = [os.path.join(train_dir, f) for train_dir in train_dir_list
                                            for f in os.listdir(train_dir) if f != "_SUCCESS"]
  #logging.info('train directory: {}'.format(train_dir))
  dataset = tf.data.TFRecordDataset(train_files)
  dataset = dataset.batch(1)
                                   
  dataset = dataset.prefetch(1) 
                                   
  iterator = dataset.make_initializable_iterator()
  next_batch = iterator.get_next()
  logging.info('train files: {}'.format(reprlib.repr(train_files)))
  
  
  with tf.Session() as sess:
    #sess.run(iterator.initializer)
    #value = sess.run(next_batch)
    meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], 'esmm_export/1590848201')
    signature = meta_graph_def.signature_def
    signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    #for n in sess.graph.as_graph_def().node:
    #  print(n.name)
    '''
    input_name = signature[signature_key].inputs['inputs'].name
    output_name = signature[signature_key].outputs['prob'].name
    input_x = sess.graph.get_tensor_by_name(input_name)
    predict_y = sess.graph.get_tensor_by_name(output_name)
    model_input = tf.saved_model.utils.build_tensor_info(input_x)
    model_output = tf.saved_model.utils.build_tensor_info(predict_y)
    batch = sess.run(predict_y, feed_dict={input_x:value})
    print(np.array(batch))
    '''
  '''
  for sec in feature_config.sections() :
    print(sec)
    info_dict = feature_config[sec]
    #if info_dict.get('ftype','')=='numeric' and ( not 'label' in sec):
    if 'label' not in sec:
      get_one(info_dict.get('fname',sec),train_inputs)
  # get tfrecods
  '''



if __name__ == '__main__':
  os.environ["CUDA_VISIBLE_DEVICES"] = ""
  logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
  parser = argparse.ArgumentParser()
  configure(parser) 
  FLAGS, _ = parser.parse_known_args()

  config = configparser.ConfigParser()  
  config.read(FLAGS.conf)

  seed = int(config['train'].get('seed', 19910825))
  tf.set_random_seed(seed)
  
  get_spec(config)
