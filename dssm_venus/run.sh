#!/bin/sh

source ~/.bashrc

source activate /data2/timmyqiu/env

alias python="CLASSPATH=$(${HADOOP_HOME}/bin/hadoop classpath --glob) python"

root_path=$(cd "$(dirname "$0")";pwd)
cd $root_path

start_date="$1"
end_date="$2"

[ ${start_date} ] ||start_date=`date -d "-8 days" +'%Y%m%d'` 
[ ${end_date} ] ||end_date=`date -d "-1 days" +'%Y%m%d'` 

p_date=${start_date}

while [ ${p_date} -lt ${end_date} ];
do
	test_date=`date -d "${p_date} +1 days" +"%Y%m%d"`
	echo "python dssm.py ${p_date} ${test_date} >logs/dssm_${test_date}.log"
	python dssm.py ${p_date} ${test_date} train >logs/dssm_${test_date}.log 2>&1
	p_date=`date -d "${p_date} +1 days" +"%Y%m%d"`
	echo "python dssm.py ${p_date} ${test_date} end."
done

