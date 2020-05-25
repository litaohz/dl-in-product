import argparse
import sys
import configparser 
#--learning_rate 20 --max_steps 10 
parse=argparse.ArgumentParser()#第一步创建ArgumentParser对象

#第二步添加可选参数
parse.add_argument("--conf",type=str,default=0.01,help="initial learining rate")

#第三步是调用parse_known_args或者parse_args解析参数，通过.参数名来访问其值
#flags,unparsed=parse.parse_known_args(sys.argv[1:])#这个也可以
flags,unparsed=parse.parse_known_args()
print("parse by parse_known_args():")
print(flags.conf)
#
print("parse by parse.args():")
args=parse.parse_args()
print(args.conf)

#读取配置文件内容 
#创建读取配置文件的对象 ConfigParser
config = configparser.ConfigParser()  
config.read(flags.conf)
#操作配置文件中的section 带中括号[]是一个个section，每个section由 key = value这种键值构成，使用字典的get方法就可以通过key取value
for sec in config.sections():
    print(sec)
    print(config[sec].get('fname', sec))
    for k,v in config[sec].items():
        print("key={},value={}".format(k,v))

