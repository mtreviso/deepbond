""" 
You must run `example_training.py` first.
"""

from deeptagger import Tagger

tagger = Tagger()
tagger.load('saved-models/testing-toy-library-mode/')

text = 'Há livros escritos para evitar espaços vazios na estante .'
classes = tagger.predict_classes(text)
probas = tagger.predict_probas(text)
tags = tagger.transform_classes_to_tags(classes)
tags_probas = tagger.transform_probas_to_tags(probas)

# you should get this:
# [4, 2, 6, 3, 2, 2, 2, 4, 2, 9]
# ['ART', 'N', 'V', 'PREP', 'N', 'N', 'N', 'ART', 'N', 'PU']
print(text)
print(classes)
print(tags)
print(tags_probas)
