from deepbond import Tagger
import numpy as np
import os
import shutil

text = 'Há livros escritos para evitar espaços vazios na estante .'

args = {
    'train_path': 'data/transcriptions/ss/Controle/',
    'dev_path': 'data/transcriptions/ss/CCL-A/',
    'test_path': 'data/transcriptions/ss/DA-Leve/',

    'model': 'rcnn',
    'rnn_type': 'gru',
    'bidirectional': True,
    'train_batch_size': 4,
    'dev_batch_size': 4,
    'optimizer': 'adam',
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

data_dir = '../../transcriptions/ss'
folds_dir = '../../folds'

if not os.path.exists(folds_dir):
    os.mkdir(folds_dir)

n_folds = 5
classes_dir = os.listdir(data_dir)

for dir in classes_dir:
    file_list = os.listdir(dir)
    np.random.shuffle(file_list)
    num_files = len(filelist)
    test_len = int(num_files / n_folds)
    fold_dir = os.path.join(folds_dir, dir)
    if not os.path.exists(fold_dir):
        os.mkdir(fold_dir)

    for id in range(4, numfiles, test_len)):
        fold_out_dir_name = os.path.join(fold_dir, str(id))
    if not os.path.exists(fold_out_dir_name):
        os.mkdir(fold_out_dir_name)
    fold_test_out = os.path.join(fold_out_dir_name, 'test')
    fold_train_out = os.path.join(fold_out_dir_name, 'train')
    for file in file_list[int(id - test_len):id]
    shutil.copyfile(os.path.join(fold_test_out), 'file2.txt')



    tagger = Tagger()
    tagger.train(dropout=0.2, ** args)
    # alternatively, you can pass them like this:
    # tagger.train(train_path='path/to/', model='rcnn', epochs=2, use_caps=True)

    classes = tagger.predict_classes(text)
    probas = tagger.predict_probas(text)
    tags = tagger.transform_classes_to_tags(classes)
    tags_probas = tagger.transform_probas_to_tags(probas)
    print(text)
    print(classes)
    print(tags)
    print(tags_probas)
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
