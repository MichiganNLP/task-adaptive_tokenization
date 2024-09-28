conda create -n mix python=3.8
conda activate mix
conda install ipython


conda install -c "nvidia/label/cuda-11.3.1" cuda-nvcc

pip install packaging sentencepiece pybind11==2.5.0 jieba tqdm nltk>=3.4 boto3==1.11.11 regex==2020.1.8 numpy>=1.15.4 pandas>=0.24.0 requests

pip install git+https://github.com/huggingface/transformers

pip install datasets 

pip install evaluate

pip install protobuf==3.20.0

pip install gdown

pip install scikit-learn