import tensorflow as tf
import argparse
import configparser
import logging
import reprlib
import sys
import os
import copy
import tensorflow as tf
from tensorflow.python.feature_column.feature_column import _LazyBuilder


def configure(parser):
  parser.add_argument(
    '--conf', type=str, help='param configuration file is requried'
)

sys.path.append("tftools/")
#sys.path.append("/data4/timmyqiu/qmkg/TensorFlowRec/tftools")
import sys
from FCGen import FCGen
from Trainer.Models.LinearModel import input_fn


parser = argparse.ArgumentParser()
configure(parser) 
FLAGS, _ = parser.parse_known_args()#解析命令行参数
config = configparser.ConfigParser()#解析配置文件用  
config.read(FLAGS.conf)

feature_config = configparser.ConfigParser()
feature_config.read(config['input']['spec'])#特征配置文件 有boundaries等信息
columns, spec = FCGen.GetFeatureSpec(feature_config)#按特征列对特征进行处理，不同类型处理会不一样，比如数值、embed等

print(spec)

def build_model_columns():
    return list(columns['user_column'].values()), list(columns['user_column'].values())
   

