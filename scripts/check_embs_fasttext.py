import numpy as np
import pickle
from gensim.models import KeyedVectors, FastText


print('Loading pickled vectors...')
f = open('../data/embs/fasttext/pt_fasttext_sg_50.small.raw.emb', 'rb')
data = pickle.load(f)
f.close()

print('Loading fasttext vectors...')
embs = FastText.load_fasttext_format('../data/embs/fasttext/pt_fasttext_sg_50.emb.bin')

words = list(data.keys())

nb_diff = 0
for word in words:
    if not np.allclose(data[word], embs[word]):
        nb_diff += 1
        print(word)

if nb_diff == 0:
    print('Everything is fine!')
else:
    print('Nb diff: {}'.format(nb_diff))
