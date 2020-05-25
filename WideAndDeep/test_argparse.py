import argparse
import sys
#--learning_rate 20 --max_steps 10 
parse=argparse.ArgumentParser()#第一步创建ArgumentParser对象

#第二步添加可选参数
parse.add_argument("--learning_rate",type=float,default=0.01,help="initial learining rate")
parse.add_argument("--max_steps",type=int,default=2000,help="max")
parse.add_argument("--hidden1",type=int,default=100,help="hidden1")

#第三步是调用parse_known_args或者parse_args解析参数，通过.参数名来访问其值
#flags,unparsed=parse.parse_known_args(sys.argv[1:])#这个也可以
flags,unparsed=parse.parse_known_args()
print("parse by parse_known_args():")
print(flags.learning_rate)
print(flags.max_steps)
print(flags.hidden1)
print(unparsed)
#
print("parse by parse.args():")
args=parse.parse_args()
print(args.learning_rate)
print(args.max_steps)
print(args.hidden1)
