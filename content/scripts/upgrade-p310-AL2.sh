# Run this commands in AWS Cloud9 terminal to upgrade Python to 3.10 for Amazon Linux 2 platform
# Install pyenv
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
cat << 'EOT' >> ~/.bashrc
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
EOT
source ~/.bashrc

# Install openSSL
sudo yum update -y
sudo yum erase openssl-devel -y
sudo yum install openssl11 openssl11-devel xz-devel libffi-devel bzip2-devel wget -y

# This command installs Python 3.10 and runs for several minutes
pyenv install 3.10
pyenv global 3.10

# Set the Python alias
export PATH="$HOME/.pyenv/shims:$PATH"

# Confirm the Python version
source ~/.bash_profile
python --version