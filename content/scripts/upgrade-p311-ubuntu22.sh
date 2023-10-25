# to see installed python versions
ls -l /usr/bin/python*

# install Python 3.11
sudo apt update -y
sudo apt install python3.11 -y

# see the default Python version
python3 --version

# set 3.11 as the default Python
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2
sudo update-alternatives --config python3

# check the default Python version
python3 --version