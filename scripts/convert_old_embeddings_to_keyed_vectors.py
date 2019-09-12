from gensim.models import Word2Vec, KeyedVectors

input_file_path = '../data/embeddings/word2vec/pt_word2vec_sg_600.emb'
output_file_path = '../data/embeddings/word2vec/pt_word2vec_sg_600.kv.emb'

print('Loading word2vec model...')
model = Word2Vec.load(input_file_path)

print('Saving embeddings as keyed vectors')
model.wv.save_word2vec_format(output_file_path)


print('Removing model from memory')
del model

print('Sanity check: loading them again using KeyedVectors class')
embeddings = KeyedVectors.load_word2vec_format(output_file_path,
                                               unicode_errors='ignore',
                                               binary=True)

print('Everything is fine!')
KeyedVectors.load()
