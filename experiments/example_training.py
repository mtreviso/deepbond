from deeptagger import Tagger

text = 'Há livros escritos para evitar espaços vazios na estante .'

args = {
  'train_path': 'data/corpus/pt/macmorpho_v1_toy/train.txt',
  'dev_path': 'data/corpus/pt/macmorpho_v1_toy/dev.txt',
  'del_word': ' ',
  'del_tag': '_',
  
  'model': 'rcnn',
  'rnn_type': 'lstm',
  'bidirectional': True,
  'train_batch_size': 128,
  'dev_batch_size': 128,
  'optimizer': 'adam',
  'final_report': True,
  'epochs': 2,
  'use_prefixes': True,
  'use_suffixes': True,
  'use_caps': True,

  'early_stopping_patience': 3,
  'restore_best_model': True,  # 

  'save_best_only': True, # only save the model
  
  'output_dir': 'runs/testing-toy-library-mode/',  # keep log information
  'save': 'saved-models/testing-toy-library-mode/',  # save models
}

tagger = Tagger()
tagger.train(**args)
# alternatively, you can pass them like this:
# tagger.train(train_path='path/to/', model='rcnn', epochs=2, use_caps=True)

classes = tagger.predict_classes(text)
probas = tagger.predict_probas(text)
tags = tagger.transform_classes_to_tags(classes)
tags_probas = tagger.transform_probas_to_tags(probas)
# you should this:
# [4, 2, 6, 3, 2, 2, 2, 4, 2, 9]
# ['ART', 'N', 'V', 'PREP', 'N', 'N', 'N', 'ART', 'N', 'PU']
print(text)
print(classes)
print(tags)
print(tags_probas)
del tagger


# loading now!

tagger = Tagger()
tagger.load('saved-models/testing-toy-library-mode/')
classes = tagger.predict_classes(text)
probas = tagger.predict_probas(text)
tags = tagger.transform_classes_to_tags(classes)
tags_probas = tagger.transform_probas_to_tags(probas)
# you should this:
# [4, 2, 6, 3, 2, 2, 2, 4, 2, 9]
# ['ART', 'N', 'V', 'PREP', 'N', 'N', 'N', 'ART', 'N', 'PU']
print(text)
print(classes)
print(tags)
print(tags_probas)
del tagger
