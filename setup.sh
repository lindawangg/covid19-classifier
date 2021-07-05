
# Remove model folder if exists
rm -rf COVIDNet-CXR-2

# Downloading model files
mkdir COVIDNet-CXR-2
wget -P COVIDNet-CXR-2/ https://github.com/saahiluppal/COVID-Net/releases/download/v0.1/checkpoint
wget -P COVIDNet-CXR-2/ https://github.com/saahiluppal/COVID-Net/releases/download/v0.1/model.data-00000-of-00001
wget -P COVIDNet-CXR-2/ https://github.com/saahiluppal/COVID-Net/releases/download/v0.1/model.index
wget -P COVIDNet-CXR-2/ https://github.com/saahiluppal/COVID-Net/releases/download/v0.1/model.meta

# install requirements
pip install -r requirements.txt
