#coding:utf-8
"""train the model"""

from optparse import OptionParser
import argparse
import configparser
import logging
import reprlib
import sys
import os
import numpy as np

import tensorflow as tf

sys.path.append("./tftools")
#sys.path.append("/data4/timmyqiu/qmkg/TensorFlowRec/tftools")
from Trainer.InputFn import input_fn

from FCGen import FCGen

def configure(parser):
  parser.add_argument(
    '--conf', type=str, help='param configuration file is requried'
  )
#参数spec_file是原始的特征配置文件，运行后会保存一份计算后得到的分桶的文件 
def  write_config(Dict,Dict2,spec_file):#第三个参数是原始的特征配置文件名
  target_file=spec_file.split('.ini')[0]+'_bucket.ini'#存放自动生成的分桶的boundary结果
  print('new_spec_name : ' + target_file)
  wf = open(target_file, 'w')
  
  field_key=''#特征配置文件中的section名，此处其实是特征名
  with open(spec_file, 'r') as fd:
    line=fd.readline()
    while line:
      if  line.startswith("["):
          field_key = line[1:-2]
          print(field_key)
      if  line.startswith("boundaries") and field_key in Dict.keys():
          line=Dict[field_key]+'\n'
      wf.write(line)
      if  line.startswith("group") and field_key in Dict2.keys():
          wf.write(Dict2[field_key]['min']+'\n')
          wf.write(Dict2[field_key]['max']+'\n')      
          wf.write(Dict2[field_key]['mean']+'\n')      
          wf.write(Dict2[field_key]['std']+'\n')      
      line = fd.readline()
  wf.close
  
def get_bucketized_bound(key,auto_num,train_inputs):
  with tf.Session() as sess:
    sess.run(train_inputs['iterator_init_op'])
    next_element = train_inputs['features']
    print(next_element)

    sample = sess.run([
        next_element[key],
    ])
    Asample=np.array(sample)
    print(Asample)
    bound='boundaries = ' + str(np.percentile(Asample,int(100/auto_num)))
    for i in range(2,auto_num):
      if(np.percentile(Asample,int(100*i/auto_num))>np.percentile(Asample,int(100*(i-1)/auto_num))):        
        bound = bound + ',' + str(np.percentile(Asample,int(100*i/auto_num)))
    print(key+":"+bound) 
    return bound    

def get_min_max(key,train_inputs):
  with tf.Session() as sess:
    sess.run(train_inputs['iterator_init_op'])
    next_element = train_inputs['features']
    print(next_element)

    sample = sess.run([
        next_element[key],
    ])
    Asample=np.array(sample)
    print(Asample)
    fmin='min = ' + str(np.percentile(Asample,1))
    fmax='max = ' + str(np.percentile(Asample,99))
    fmean='mean = ' + str(np.mean(Asample))
    fstd='std = ' + str(np.std(Asample))

    dicts={}
    dicts['min']=fmin
    dicts['max']=fmax
    dicts['mean'] = fmean
    dicts['std'] = fstd

    return dicts  
    
        
def write_spec(config):
  """Entry for trainig

  Args:
    config: (configparser) All the hyperparameters for training
  """
  #注意计算分桶的边界使用一天的样本就可以不需要多天
  train_dir = config['input']['train']
  train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f != "_SUCCESS"]
  logging.info('train directory: {}'.format(train_dir))
  logging.info('train files: {}'.format(reprlib.repr(train_files)))

  feature_config = configparser.ConfigParser()
  feature_config.read(config['input']['spec'])#读取特征原始配置文件，这里存放特征名，需要的分桶数等信息
  columns, spec = FCGen.GetFeatureSpec(feature_config)

  logging.info("Creating iterators...")
  batch_size = int(config['train']['batch_size'])
  # 调用Trainer.InputFn的input_fn 使用tf.data.TFRecordDataset接口来读取数据
  train_inputs = input_fn(train_files, spec, shuffle=True, batch_size=100000)

  # get tfrecods
  num_epochs = 1
  logging.info("Start training for {} epoch(s)".format(num_epochs))

  dict_liner = {}#linear部分特征计算boundaries,保存为字典：{feat_name:boundaries}
  for sec in feature_config.sections():#遍历配置文件，读取各个特征需要的分桶数量
    print(sec)
    info_dict = feature_config[sec]
    auto_num = int(info_dict.get('auto_boundaries', 0))
    if auto_num > 0 :
      dict_liner[sec]=get_bucketized_bound(info_dict.get('fname', sec),auto_num,train_inputs)#info_dict.get('fname', sec)获取的还是sec

  dict_deep = {'min':{},'max':{}}#deep部分特征是计算最大值、最小值、均值、std,结构是字典的嵌套，{feat_ename:{"max":xxx,"min":xxx}}
  for sec in feature_config.sections() :
    print(sec)
    info_dict = feature_config[sec]
    groups = info_dict.get('group','').split(',')
    if 'norm' in groups and  info_dict.get('ftype','')=='numeric' and sec!='label':
      dict_deep[sec]=get_min_max(info_dict.get('fname',sec),train_inputs)
      
  write_config(dict_liner,dict_deep,config['input']['spec']) 


if __name__ == '__main__':
  
  #python spec_bucket.py --conf conf.ini
  os.environ["CUDA_VISIBLE_DEVICES"] = "5"
  logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
  parser = argparse.ArgumentParser()
  configure(parser) 
  FLAGS, _ = parser.parse_known_args()
  print(FLAGS)

  config = configparser.ConfigParser()  
  config.read(FLAGS.conf)

  seed = int(config['train'].get('seed', 19910825))
  tf.set_random_seed(seed)

  write_spec(config)
