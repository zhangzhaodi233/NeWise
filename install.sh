#!/bin/bash

sudo apt update
echo "Y" | sudo apt upgrade -y
sudo apt install build-essential 
sudo apt install zlib1g-dev 
echo "Y" | sudo apt install libbz2-dev
echo "Y" | sudo apt install libncurses5-dev
sudo apt install libgdbm-dev 
echo "Y" | sudo apt install libnss3-dev
sudo apt install libssl-dev 
sudo apt install libreadline-dev 
sudo apt install libffi-dev
sudo apt-get install -y libgl1-mesa-dev
sudo apt-get install wget

wget https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz
tar -xzvf Python-3.7.5.tgz
cd Python-3.7.5
./configure --prefix=/usr/local/src/python37
sudo make
sudo make install
sudo ln -s /usr/local/src/python37/bin/python3.7 /usr/bin/python3.7
sudo ln -s /usr/local/src/python37/bin/pip3.7 /usr/bin/pip3.7
cd ../

echo "Y" | sudo apt-get install python-virtualenv
virtualenv -p python3.7 venv
source venv/bin/activate
pip install -r requirements.txt

python modify_file.py

