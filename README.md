original
- https://github.com/r9y9/deepvoice3_pytorch


nvidia-docker run --cpu-shares=2048 --shm-size 50G -it -d -v /mnt/tts/:/workspace -w /workspace --name pytorch_tts -e LC_ALL=C.UTF-8 --entrypoint=/bin/bash pytorch/pytorch:1.4-cuda10.1-cudnn7-devel


TTS deepvoice3_pytorch 설치(wavenet-vocoder branch까지 되게)
###
git clone https://github.com/r9y9/deepvoice3_pytorch 또는
git clone -b wavenet-vocoder https://github.com/r9y9/deepvoice3_pytorch.git
###
cd /workspace/deepvoice3_pytorch
apt-get update
apt-get install -y vim libc6-dev gcc g++ libsndfile1 wget python3-pyqt5 make
pip install -e ".[bin]"
pip install docopt tensorflow-gpu==1.15.0 nnmnkwii matplotlib tensorboardX PyQt5 pybind11 python-mecab-ko g2pk wavenet_vocoder
python -c "import nltk; nltk.download('cmudict')"
python -c "import nltk; nltk.download('punkt')"
