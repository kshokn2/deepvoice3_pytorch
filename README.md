<h2>original source</h2>
- https://github.com/r9y9/deepvoice3_pytorch
</br></br>

<h2>docker</h2>
nvidia-docker run --cpu-shares=2048 --shm-size 50G -it -d -v /mnt/tts/:/workspace -w /workspace --name pytorch_tts -e LC_ALL=C.UTF-8 --entrypoint=/bin/bash pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
</br></br>

<h2>installation(wavenet-vocoder branch까지 되게)</h2>
<pre>
git clone https://github.com/r9y9/deepvoice3_pytorch 또는 git clone -b wavenet-vocoder https://github.com/r9y9/deepvoice3_pytorch.git
cd /workspace/deepvoice3_pytorch
apt-get update
apt-get install -y vim libc6-dev gcc g++ libsndfile1 wget python3-pyqt5 make
pip install -e ".[bin]"
pip install docopt tensorflow-gpu==1.15.0 nnmnkwii matplotlib tensorboardX PyQt5 pybind11 python-mecab-ko g2pk wavenet_vocoder
python -c "import nltk; nltk.download('cmudict')"
python -c "import nltk; nltk.download('punkt')"
</pre>
</br></br>

<h2>data downloading(son) and preprocessing with json</h2>
adjust trim
ref: https://github.com/carpedm20/multi-speaker-tacotron-tensorflow
</br></br>

<h2>synthesize</h2>
korean multi-speaekr and single-speaker
<h3>kss(multi_speaker)</h3>
https://drive.google.com/drive/folders/1tfRHz813VcnNO6pcelzKnldcAgGz3qdu?usp=sharing
</br>
<h3>son(single_speaker)</h3>
https://drive.google.com/drive/folders/1qG2YQC0QHIeGDouHFlm9bI7kU2W2dtA8?usp=sharing
