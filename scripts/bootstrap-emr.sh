#!/bin/bash

set -e
# Record starting time
touch $HOME/.bootstrap-begin

sudo yum -y update
sudo yum -y install tmux

# Create the anaconda directory on a volume with more space
#mkdir -p /home/hadoop/anaconda
sudo mkdir /mnt/anaconda
sudo chown hadoop:hadoop /mnt/anaconda
ln -s /mnt/anaconda $HOME/anaconda

# Create the nltk_data directory on a volume with more space
#mkdir -p /home/hadoop/nltk_data
sudo mkdir /mnt/nltk_data
sudo chown hadoop:hadoop /mnt/nltk_data
ln -s /mnt/nltk_data $HOME/nltk_data

touch $HOME/msg
echo "Downloading Anaconda\n" >> $HOME/msg

# Download Anaconda
sudo wget -S -T 10 -t 5 https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O $HOME/anaconda/anaconda.sh

# Install Anaconda
sudo bash $HOME/anaconda/anaconda.sh -u -b -p $HOME/anaconda

# Add Anaconda to current session's PATH
export PATH=$HOME/anaconda/bin:$PATH

# Add Anaconda to PATH for future sessions via .bashrc
echo -e "\n\n# Anaconda" >> $HOME/.bashrc
echo "export PATH=$HOME/anaconda/bin:$PATH" >> $HOME/.bashrc

cd $HOME

export PYSPARK_PYTHON=/home/hadoop/anaconda/bin/python
export PYSPARK_DRIVER_PYTHON=/home/hadoop/anaconda/bin/python

echo "Finished Setting Anaconda Env" >> $HOME/msg

#$HOME/anaconda/bin pip install boto3
#echo "Finished installing boto3" >> $HOME/msg

aws s3 cp s3://s3-aka-emr-cluster/src/main.py $HOME/main.py
aws s3 cp s3://s3-aka-emr-cluster/src/pipeline.py $HOME/pipeline.py
aws s3 cp s3://s3-aka-emr-cluster/src/model.py $HOME/model.py
aws s3 cp s3://s3-aka-emr-cluster/scripts/config_spark_env $HOME/config_spark_env
aws s3 cp s3://s3-aka-emr-cluster/data/Tools_and_Home_Improvement_5.json $HOME/Tools_and_Home_Improvement_5.json
echo "Finished loading source and data files from s3 bucket" >> $HOME/msg

$HOME/anaconda/bin/python -m pip install boto3

# Download NLTK libraries
/home/hadoop/anaconda/bin/python -c "import nltk; \
nltk.download('stopwords', '$HOME/nltk_data'); \
nltk.download('punkt', '$HOME/nltk_data'); \
nltk.download('wordnet', '$HOME/nltk_data'); \
nltk.download('averaged_perceptron_tagger', '$HOME/nltk_data'); \
nltk.download('maxent_treebank_pos_tagger', '$HOME/nltk_data')"

echo "Finished Downloading NLTK lib" >> $HOME/msg

#cd /home/hadoop
# Record ending time
touch $HOME/.bootstrap-end