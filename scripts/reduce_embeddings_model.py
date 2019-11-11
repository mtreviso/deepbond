import argparse
import pickle
import re
from pathlib import Path

from gensim.models import KeyedVectors, FastText


def normalize(text):
    t = text.strip()
    t = re.sub(r'(\S+)([\.\,\?\!\:\;\*])', r'\1 \2', t.strip())
    t = re.sub(r'([\.\,\?\!\:\;\*])(\S+)', r'\1 \2', t.strip())
    t = re.sub(r'\ +', ' ', t.strip())
    return t


def get_vocab(data_path):
    vocab = set()
    root_dir_path = Path(data_path)
    for dir_path in root_dir_path.iterdir():
        for f_path in dir_path.iterdir():
            with f_path.open('r', encoding='utf8') as f:
                paragraph = normalize(f.read())
                words = paragraph.split()
                vocab.update(set(words))
    return vocab


def has_number(word):
    return any(char.isdigit() for char in word)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Reduce embeddings model")
    parser.add_argument("--emb-path",
                        type=str,
                        default='../data/embs/word2vec/pt_word2vec_sg_600.kv.emb',
                        help="path to keyed vector embedding model")
    parser.add_argument("--data-path",
                        type=str,
                        default='../data/transcriptions/ss/',
                        help="path to the dataset dirs: CCL-A, Controle, DA-Leve")
    parser.add_argument("--output-path",
                        type=str,
                        default='../data/embs/word2vec/pt_word2vec_sg_600.small.raw.emb',
                        help="path to the new embeddings")
    parser.add_argument('--binary',
                        action='store_true',
                        help='Whether to save the embeddings are in binary format or not.')
    parser.add_argument("-f", "--format",
                        type=str,
                        default="word2vec",
                        choices=['word2vec', 'fasttext'],
                        help="embeddings format")
    args = parser.parse_args()

    vocab = get_vocab(args.data_path)
    print('Vocab size: {}'.format(len(vocab)))

    if args.format == 'word2vec':
        embeddings = KeyedVectors.load_word2vec_format(args.emb_path,
                                                       unicode_errors='ignore',
                                                       binary=True)
    else:
        embeddings = FastText.load_fasttext_format(args.emb_path)

    word_vectors = {}
    for word in vocab:
        if word in embeddings:
            word_vectors[word] = embeddings[word]
        else:
            print('{} not found in model vocab. It will be replaced later '
                  'by an unknown vector.'.format(word))

    if args.binary:
        with open(args.output_path, 'wb') as handle:
            pickle.dump(word_vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        f = open(args.output_path, 'w', encoding='utf8')
        for word, vector in word_vectors.items():
            s = ' '.join([word] + [str(d) for d in vector])
            f.write(s + '\n')
        f.close()

