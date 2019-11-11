"""
For FastText only:
---
.vec contains ONLY word-vectors (no ngrams here), can be loaded with KeyedVectors.load_word2vec_format
.bin contains ngrams, can be loaded with FastText.load_fasttext_format
---
So, pass the .bin file and you will be fine here! But,
After transforming to KeyedVectors, FastText is not able to get ngrams vectors anymore!
"""

import argparse

# pip3 install --user gensim
from gensim.models import Word2Vec, KeyedVectors, FastText


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Reduce embeddings model")
    parser.add_argument("-i", "--input-path",
                        type=str,
                        default='../data/embs/word2vec/pt_word2vec_sg_600.emb',
                        help="path to the old vector embedding model")
    parser.add_argument("-o", "--output-path",
                        type=str,
                        default='../data/embs/word2vec/pt_word2vec_sg_600.kv.emb',
                        help="path to the new embeddings. ")
    parser.add_argument("-f", "--format",
                        type=str,
                        default="word2vec",
                        choices=['word2vec', 'fasttext'],
                        help="embeddings format")
    args = parser.parse_args()

    input_file_path = args.input_path
    output_file_path = args.output_path

    print('Loading model...')
    if args.format == 'word2vec':
        model = Word2Vec.load(input_file_path)
    else:
        print('After transforming to KeyedVectors, fasttext is not able to '
              'get ngrams vectors anymore! Be careful.')
        model = FastText.load_fasttext_format(input_file_path)

    print('Cinderela embedding with mean and std: ')
    print(model['cinderela'].mean(), model['cinderela'].std())

    print('Saving embeddings as keyed vectors')
    model.wv.save_word2vec_format(output_file_path, binary=True)

    print('Removing model from memory')
    del model

    print('Sanity check: loading them again using KeyedVectors class')
    embeddings = KeyedVectors.load_word2vec_format(output_file_path,
                                                   unicode_errors='ignore',
                                                   binary=True)

    print('Cinderela embedding with mean and std: ')
    print(embeddings['cinderela'].mean(), embeddings['cinderela'].std())

    print('Everything is fine!')
