#coding:utf-8
"""train the model"""

import argparse
import configparser
import logging
import reprlib
import sys
import os
import copy
import tensorflow as tf
from esmm import esmm_model_fn
from tensorflow.python.feature_column.feature_column import _LazyBuilder
sys.path.append("tftools/")
#sys.path.append("/data4/timmyqiu/qmkg/TensorFlowRec/tftools")

from FCGen import FCGen
from Trainer.Models.LinearModel import input_fn

def configure(parser):
  parser.add_argument(
    '--conf', type=str, help='param configuration file is requried'
  )
  '''
  parser.add_argument(
    '--train', type=str, help='param configuration file is requried'
  )
  parser.add_argument(
    '--dev', type=str, help='param configuration file is requried'
  )
  '''

def input_receiver(feature_spec):
  serialized_tf_example = tf.placeholder(dtype=tf.string,
                                         shape=[None],
                                         name='input')
  receiver_tensors = {'inputs': serialized_tf_example}
  features = tf.parse_example(serialized_tf_example, feature_spec)
  rec = tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
  print('rec:', rec)
  return rec  

def metric_auc(labels, predictions):
  return {
    'auc': tf.metrics.auc(labels=labels, predictions=predictions['logistic'], summation_method="careful_interpolation")
  }

def get_norm_keys(conf_path):
  keys = set()
  f = open(conf_path)
  for line in f:
    keys.add(line.strip())
  f.close()
  return keys

def adjust_features_columns(keys, linear_columns):
  for key in keys:
    if key in linear_columns:
      linear_columns.pop(key)

#config：配置文件传入参数
def train(config , trainfile, testfile):
  """Entry for trainig

  Args:
    config: (configparser) All the hyperparameters for training
  """
  keys = get_norm_keys(config['input'].get('conf'))
  train_dirs = trainfile.split(',')
  train_files = [[] for _ in range(5)]
  for train_dir in train_dirs:
    for f in os.listdir(train_dir):
      if f != "_SUCCESS":
        ind = int(int(f.split('-')[-1]) / 40)
        train_files[ind].append(os.path.join(train_dir, f))
  #train_files = [os.path.join(train_dir, f) for train_dir in train_dirs for f in os.listdir(train_dir) if f != "_SUCCESS"]
  #train_files = tf.random_shuffle(tf.train.match_filenames_once([os.path.join(train_dir, f) for f in os.listdir(train_dir) if f != "_SUCCESS"]))
  #train_files = tf.random_shuffle(tf.train.match_filenames_once(['%s/%s/part-r-*' % (data_path, dt) for dt in date_list]))
  logging.info('train directory: {}'.format(train_dirs))
  logging.info('train files: {}'.format(reprlib.repr(train_files)))

  dev_dirs = testfile.split(',')
  dev_files = [os.path.join(dev_dir, f) for dev_dir in dev_dirs for f in os.listdir(dev_dir) if f != "_SUCCESS"]
  logging.info('dev directory: {}'.format(dev_dirs))
  logging.info('dev files: {}'.format(reprlib.repr(dev_files)))
  #特征的配置文件 在input 这个section的spec这个key
  feature_config = configparser.ConfigParser()
  feature_config.read(config['input']['spec'])#特征配置文件 有boundaries等信息
  columns, spec = FCGen.GetFeatureSpec(feature_config)#按特征列对特征进行处理，不同类型处理会不一样，比如数值、embed等
  
  batch_size = int(config['train']['batch_size'])
  
  conf = tf.ConfigProto()  
  conf.gpu_options.allow_growth=True  

  os.environ["CUDA_VISIBLE_DEVICES"] = "5"
  run_config = tf.estimator.RunConfig().replace(
    model_dir=config['train'].get('model_dir', 'model_dir'),  
    session_config=conf)
  
  logging.info("Creating model...")
  # Define the model
  hidden_units = [int(n) for n in config['model']['hidden_units'].split(',')]
  learning_rate = float(config['model']['learning_rate'])
  ctr_reg = float(config['model'].get('ctr_reg', '1e-6'))
  cvr_reg = float(config['model'].get('cvr_reg', '1e-4'))
  ctcvr_loss_weight = float(config['model'].get('ctcvr_loss_weight', '1.0'))
  model = tf.estimator.Estimator(
    model_fn=esmm_model_fn,
    params={
      'dnn_columns': list(columns['dnn'].values()),
      'linear_columns': list(columns['linear'].values()),
      'weight_columns': list(columns['weight'].values()),
      'hidden_units': hidden_units,
      'learning_rate': learning_rate,
      'ctr_reg': ctr_reg,
      'cvr_reg': cvr_reg,
      'ctcvr_loss_weight': ctcvr_loss_weight,
      'model': config['model']['model']
    },
    config = run_config
  )
  print(model.evaluate(input_fn=lambda: input_fn(dev_files[0:1], spec, False, batch_size, mt=True)))
  model.export_savedmodel(export_dir_base=config['train'].get('export_dir', 'export_dir'), 
      serving_input_receiver_fn=lambda: input_receiver(spec),
      strip_default_attrs=True) 

if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
  parser = argparse.ArgumentParser()
  configure(parser) 
  FLAGS, _ = parser.parse_known_args()#解析命令行参数

  config = configparser.ConfigParser()#解析配置文件用  
  config.read(FLAGS.conf)

  seed = int(config['train'].get('seed', 19910825))
  tf.set_random_seed(seed)
  train_dir = config['input'].get('train')
  test_dir = config['input'].get('dev')
  #train(config ,FLAGS.train , FLAGS.dev)
  train(config, train_dir, test_dir)
