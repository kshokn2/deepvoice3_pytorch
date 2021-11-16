'''
This script is made for trimming leading and trailing silence from an audio signal.
Especially in NIKL dataset for korean tts, I'm supposed to trim the all audio files.

usage:
    python trim_slience_in_wav.py --in_dir=path/to/input/dir --out_dir=output/path/
'''

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', type=str, required=True)
parser.add_argument('--out_dir', type=str, required=True)
args = parser.parse_args()

name = args.in_dir #'/Users/mz02-ksh/to_aws/fv01'
for n in os.listdir(name):
    f = os.path.join(name,n)
    y, sr = librosa.load(f)
    
    # trim
    yt, _ = librosa.effects.trim(y, top_db=30);print(f,len(y),len(yt))
    sf.write(args.out_dir, yt.T, sr, 'PCM_16')

