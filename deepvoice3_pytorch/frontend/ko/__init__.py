# coding: utf-8

from nltk import word_tokenize
#from deepvoice3_pytorch.frontend.ko.KoG2P.g2p import runKoG2P
from deepvoice3_pytorch.frontend.text.symbols import symbols

from random import random

types = 1
use_g2pk = True

if types == 1:
    # original script
    n_vocab = 0xffff

    _eos = 1
    _pad = 0
    if use_g2pk and None:
        from g2pk import G2p
        g2p = G2p()
    else:
        g2p = None

elif types == 2:
    # 1. KoG2P (https://github.com/scarletcho/KoG2P)
    _eos = '~'
    _pad = '_'
    ONS = ['k0', 'kk', 'nn', 't0', 'tt', 'rr', 'mm', 'p0', 'pp',
            's0', 'ss', 'oh', 'c0', 'cc', 'ch', 'kh', 'th', 'ph', 'h0']
    NUC = ['aa', 'qq', 'ya', 'yq', 'vv', 'ee', 'yv', 'ye', 'oo', 'wa',
            'wq', 'wo', 'yo', 'uu', 'wv', 'we', 'wi', 'yu', 'xx', 'xi', 'ii']
    COD = ['', 'kf', 'kk', 'ks', 'nf', 'nc', 'nh', 'tf',
            'll', 'lk', 'lm', 'lb', 'ls', 'lt', 'lp', 'lh',
            'mf', 'pf', 'ps', 's0', 'ss', 'oh', 'c0', 'ch',
            'kh', 'th', 'ph', 'h0']
    symbols = ONS + NUC + COD + list('!\'(),-.:;? ') + ['ng', '\\1'] + [_pad, _eos]
    n_vocab = len(symbols)


_tagger = None

# test_2
def text_to_sequence(text, p=0.0, as_token=False):
    from deepvoice3_pytorch.frontend.ko.test_t2s import text_to_sequence
    #if use_g2pk and g2p is not None:
    #    text = g2p(text)
    text = text.replace('\n', '')#.replace('.', ' ')
    text = text if text[-1] == '.' or text[-1] == '!' or text[-1] == '?' else text + '.'
    text = text_to_sequence(text, as_token)
    return text

'''
# test_1
def text_to_sequence(text, p=0.0):
    if types == 1:
        if use_g2pk and g2p is not None:
            text = g2p(text)
        new_text = text.replace('\n', '').replace('.', ' ')
        return [ord(c) for c in new_text] + [_eos]  # EOS

    elif types == 2:
        word = text.replace('\n','').split(' ') #word_tokenize(text)
        seq = []
        for w in word:
            seq += runKoG2P(w, './deepvoice3_pytorch/frontend/ko/KoG2P/rulebook.txt').split(' ')
        seq += [_eos]
        return [symbols.index(s) for s in seq]
'''

def sequence_to_text(seq):
    return "".join(chr(n) for n in seq)
