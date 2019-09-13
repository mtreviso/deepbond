from deepbond import Tagger

import os

text = 'Há livros escritos para evitar espaços vazios na estante .'

args = {
    'model': 'rcnn',
    'rnn_type': 'lstm',
    'bidirectional': True,
    'train_batch_size': 4,
    'dev_batch_size': 4,
    'optimizer': 'rmsprop',
    'final_report': True,
    'epochs': 1,
    'hidden_size': [100],

    'punctuations': '.?!',
    'loss_weights': 'balanced',
    # 'embeddings_format': 'word2vec',
    # 'embeddings_path': 'data/embeddings/word2vec/pt_word2vec_sg_600.kv.emb',

    'early_stopping_patience': 5,
    'restore_best_model': True,  #

    'save_best_only': True,  # only save the best model

    'output_dir': 'runs/testing-cinderela-lib-mode/',  # keep log information
    'save': 'saved-models/testing-cinderela-lib-mode/',  # save models
    'tensorboard': True,

}

folds_dir = 'data/transcriptions/folds/CCL-A/'

for fold in os.listdir(folds_dir):
    print("Fold Number:", fold)
    train_path = os.path.join(folds_dir, fold, 'train')
    test_path = os.path.join(folds_dir, fold, 'test')

    tagger = Tagger()
    tagger.train(train_path=train_path,
                 test_path=test_path,
                 dropout=0.2,
                 **args)

    # alternatively, you can pass them like this:
    # tagger.train(train_path='path/to/', model='rcnn', epochs=2, use_caps=True)

    classes = tagger.predict_classes(text)
    tags = tagger.transform_classes_to_tags(classes)

    print(text)
    print(tags)
    del tagger

# loading now!
tagger = Tagger()
tagger.load('saved-models/testing-cinderela-lib-mode/')
classes = tagger.predict_classes(text)
probas = tagger.predict_probas(text)
tags = tagger.transform_classes_to_tags(classes)
tags_probas = tagger.transform_probas_to_tags(probas)
print(text)
print(classes)
print(tags)
print(tags_probas)
del tagger
