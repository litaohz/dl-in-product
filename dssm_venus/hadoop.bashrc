# .bashrc

# User specific aliases and functions

#alias rm='rm -i'
#alias cp='cp -i'
#alias mv='mv -i'
alias vi='vim'

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# added by Anaconda3 installer
#export PATH="/root/anaconda3/bin:$PATH"

export PATH="/root/driver/hadoop-0.20.1/bin:/usr/local/cuda-9.0/bin:/root/anaconda3/bin:/usr/local/bin:$PATH"
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda-9.0/lib64:/usr/local/cuda-9.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-9.0/

export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.71-2.b15.el7_2.x86_64/jre/
export LD_LIBRARY_PATH=${JAVA_HOME}/lib/amd64/server:${LD_LIBRARY_PATH}

export HADOOP_HDFS_HOME="/data/tdwdfsclient/"
export HADOOP_HOME="/data/tdwdfsclient/"
export PATH="${HADOOP_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH=${HADOOP_HOME}/lib/native:${LD_LIBRARY_PATH}
source ${HADOOP_HOME}/libexec/hadoop-config.sh

alias tdfs="hadoop fs -Dfs.defaultFS=hdfs://ss-sng-dc-v2 -Dhadoop.job.ugi=tdw_felixzzhao:zhaofeng1008" 
alias tdfs_su="hadoop fs -Dfs.defaultFS=hdfs://ss-sng-dc-v2 -Dhadoop.job.ugi=tdwadmin:supergroup" 
export HDFS_OUT_PATH=/stage/outface/ecc/u_ecc_qqmusicaudio
export HDFS_IN_PATH=/stage/interface/ecc/u_ecc_qqmusicaudio

alias python="CLASSPATH=$(${HADOOP_HOME}/bin/hadoop classpath --glob) python"

