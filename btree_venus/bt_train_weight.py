#coding:utf-8
"""train the model"""

import argparse
import configparser
import logging
import reprlib
import sys
import os
import json
import tensorflow as tf
import datetime

sys.path.append("./tftools/")
#sys.path.append("/data4/timmyqiu/qmkg/TensorFlowRec/tftools")

from FCGen import FCGen
from Trainer.Models.LinearModel import input_fn_pattern as input_fn

def configure(parser):
  parser.add_argument(
    '--conf', type=str, help='param configuration file is requried'
  )

def metric_auc(labels, predictions):
  return {
    'auc': tf.metrics.auc(labels=labels, predictions=predictions['logistic'], summation_method="careful_interpolation")
  }

def input_receiver(feature_spec):
  serialized_tf_example = tf.placeholder(dtype=tf.string,
                                         shape=[None],
                                         name='input')
  receiver_tensors = {'inputs': serialized_tf_example}
  features = tf.parse_example(serialized_tf_example, feature_spec)
  rec = tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
  print('rec:', rec)
  return rec  
  
#config：配置文件传入参数
def train(config , hdfs_prefix, ftime, gap, ckpt_dir, export_dir, metric_dir):
  """Entry for trainig

  Args:
    config: (configparser) All the hyperparameters for training
  """
  train_files = []
  dev_files = []
  cur_date = datetime.datetime.strptime(ftime, "%Y%m%d")
  for i in range(1, gap+1):
    dest_date = (cur_date + datetime.timedelta(days=-i)).strftime("%Y%m%d")
    train_files.append(hdfs_prefix + "/" + dest_date + "/train/part-r-*")
    dev_files.append(hdfs_prefix + "/" + dest_date + "/test/part-r-*")
  logging.info('train files: {}'.format(reprlib.repr(train_files)))
  
  logging.info('dev files: {}'.format(reprlib.repr(dev_files)))
  #特征的配置文件 在input 这个section的spec这个key
  feature_config = configparser.ConfigParser()
  feature_config.read(config['input']['spec'])#特征配置文件 有boundaries等信息
  columns, spec = FCGen.GetFeatureSpec(feature_config)#按特征列对特征进行处理，不同类型处理会不一样，比如数值、embed等

  batch_size = int(config['train']['batch_size'])

  conf = tf.ConfigProto()  
  conf.gpu_options.allow_growth=True

  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  run_config = tf.estimator.RunConfig(
      save_checkpoints_secs=1800).replace(
      session_config=conf)

  
  logging.info("Creating model...")
  # Define the model
  model = tf.estimator.BoostedTreesClassifier(
            config=run_config,
            n_batches_per_layer=1000,
            n_trees=100,
            learning_rate=0.2,
            l1_regularization=0.01,
            l2_regularization=0.01,
            max_depth=10, 
            model_dir=ckpt_dir,
            feature_columns=list(columns['linear'].values()),
            weight_column=list(columns['weight'].values())[0]#如果有weight这个就不能注释
          )
  #model = tf.estimator.add_metrics(model, metric_auc)
  # Train and evaluate
  epochs = int(config['train']['epochs'])
  for i in range(epochs):
    logging.info("training...")
    model.train(input_fn=lambda: input_fn(train_files, spec, shuffle=True, batch_size=batch_size))

    results = model.evaluate(input_fn=lambda: input_fn(dev_files, spec, shuffle=False, batch_size=batch_size))
    auc = float(results["auc"])
    logloss = float(results["loss"])
  index = [{"name": "auc", "type": "float", "value": str(auc)}, {"name": "logloss", "type": "float", "value": str(logloss)}]
  file_name = metric_dir + "/metrics_info.json"
  with open(file_name, 'w') as file_obj:
    json.dump(index, file_obj)
  model.export_savedmodel(export_dir_base=export_dir,
      serving_input_receiver_fn=lambda: input_receiver(spec),
      strip_default_attrs=True)
 
if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
  parser = argparse.ArgumentParser()
  configure(parser) 
  FLAGS, _ = parser.parse_known_args()#解析命令行参数

  config = configparser.ConfigParser()#解析配置文件用  
  config.read(FLAGS.conf)
  ftime = sys.argv[2]
  ckpt_dir = sys.argv[3]
  export_dir = sys.argv[4]
  metric_dir = sys.argv[5]
  gap = int(sys.argv[6])
  seed = int(config['train'].get('seed', 19910825))
  tf.set_random_seed(seed)
  hdfs_path = config['input'].get('hdfs_path')
  print("checkpoint_dir:", ckpt_dir)
  train(config, hdfs_path, ftime, gap, ckpt_dir, export_dir, metric_dir)
