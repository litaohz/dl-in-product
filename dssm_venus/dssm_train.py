#coding:utf-8
"""train the model"""
time = __import__('time')
datetime = __import__('datetime')
import argparse
import configparser
import logging
import reprlib
import sys
import os
import copy
import tensorflow as tf
from dssm_v1 import dssm_model_fn
from tensorflow.python.feature_column.feature_column import _LazyBuilder
sys.path.append("tftools/")
#sys.path.append("/data4/timmyqiu/qmkg/TensorFlowRec/tftools")

from FCGen import FCGen
from Trainer.Models.LinearModel import input_fn_pattern


def get_date_delta(baseTime, delta):
    # baseTime： is a string like "19700101"
    # delta：   1 later, -1: before
    d1 = datetime.datetime(int(baseTime[0:4]), int(baseTime[4:6]), int(baseTime[6:8]))
    d2 = d1 + datetime.timedelta(days=delta)
    deltaDate = d2.strftime("%Y%m%d")
    return deltaDate

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
def train(config , trainfile, testfile, time):
  """Entry for trainig

  Args:
    config: (configparser) All the hyperparameters for training
  """
  #keys = get_norm_keys(config['input'].get('conf'))
  #print("get_norm_keys")
  #print(keys)
  train_dirs = trainfile.split(',')
  train_files = [train_dir + 'part-r-*' for train_dir in train_dirs]
  #train_files = tf.random_shuffle(tf.train.match_filenames_once([os.path.join(train_dir, f) for f in os.listdir(train_dir) if f != "_SUCCESS"]))
  #train_files = tf.random_shuffle(tf.train.match_filenames_once(['%s/%s/part-r-*' % (data_path, dt) for dt in date_list]))
  logging.info('train directory: {}'.format(train_dirs))
  logging.info('train files: {}'.format(reprlib.repr(train_files)))

  dev_dirs = testfile.split(',')
  dev_files = [dev_dir + 'part-r-*' for dev_dir in dev_dirs]
  logging.info('dev directory: {}'.format(dev_dirs))
  logging.info('dev files: {}'.format(reprlib.repr(dev_files)))
  #特征的配置文件 在input 这个section的spec这个key
  feature_config = configparser.ConfigParser()
  
  feature_config.read(config['input']['spec'])#特征配置文件 有boundaries等信息

  columns, spec = FCGen.GetFeatureSpec(feature_config)#按特征列对特征进行处理，不同类型处理会不一样，比如数值、embed等
  print(columns.keys())
  print("++++++user_columns")
  for v in columns['user_columns']:
    print(v)
  print("++++++item_columns")
  for v in columns['item_columns']:
    print(v)
    
  batch_size = int(config['train']['batch_size'])
  
  conf = tf.ConfigProto()  
  conf.gpu_options.allow_growth=True  
  model_dir = config['train'].get('model_dir', 'model_dir') + time + "/"
  os.environ["CUDA_VISIBLE_DEVICES"] = "5"
  run_config = tf.estimator.RunConfig().replace(
    #model_dir=config['train'].get('model_dir', 'model_dir') + '_' + time + "/",  
    model_dir=model_dir,
    session_config=conf)
  train_input_fn = lambda: input_fn_pattern(train_files, spec, True, batch_size)
  eval_input_fn = lambda: input_fn_pattern(dev_files, spec, False, batch_size)
  
  logging.info("Creating model...")
  # Define the model
  hidden_units = [int(n) for n in config['model']['hidden_units'].split(',')]
  learning_rate = float(config['model']['learning_rate'])
  ctr_reg = float(config['model'].get('ctr_reg', '1e-6'))
  cvr_reg = float(config['model'].get('cvr_reg', '1e-4'))
  ctcvr_loss_weight = float(config['model'].get('ctcvr_loss_weight', '1.0'))
  model = tf.estimator.Estimator(
    model_fn=dssm_model_fn,
    params={
      'user_columns': list(columns['user_columns'].values()),
      'item_columns': list(columns['item_columns'].values()),
      'hidden_units': hidden_units,
      'learning_rate': learning_rate,
      'ctr_reg': ctr_reg,
      'cvr_reg': cvr_reg,
      'ctcvr_loss_weight': ctcvr_loss_weight,
      'model': config['model']['model'],
      'embedding_size': int(config['model'].get('embedding_size', '4'))
    },
    config = run_config
  )
  # Train and evaluate
  max_steps = config['train'].get('max_step', '')
  if max_steps == '':
    max_steps = None
  else:
    max_steps = int(max_steps)

  logging.info("training...")
  epochs = int(config['train'].get('epochs', '1'))
  for i in range(epochs):
    logging.info("{}th training...".format(i+1))
    model.train(input_fn=train_input_fn, steps=max_steps)

    results = model.evaluate(input_fn=eval_input_fn, steps=max_steps)

    logging.info("{}th results...".format(i+1))
    for key in sorted(results):
      print('%s: %s' % (key, results[key]))
  export_dir_base = config['train'].get('export_dir', 'export_dir') + time + "/"
  model.export_savedmodel(export_dir_base=export_dir_base,
      serving_input_receiver_fn=lambda: input_receiver(spec),
            strip_default_attrs=True) 

if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
  parser = argparse.ArgumentParser()
  configure(parser) 
  FLAGS, _ = parser.parse_known_args()#解析命令行参数

  config = configparser.ConfigParser()#解析配置文件用  
  config.read(FLAGS.conf)
  time = sys.argv[2]
  print("dssm begin at:" + time)
  seed = int(config['train'].get('seed', 19910825))
  tf.set_random_seed(seed)
  train_dir = config['input'].get('train') + time + "/train/" 
  test_dir = config['input'].get('dev')+ time + "/test/"
  #train(config ,FLAGS.train , FLAGS.dev)
  train(config, train_dir, test_dir, time)
