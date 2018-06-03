# deepbondd
Deep neural approach to Boundary and Disfluency Detection

This is part of my MSc project. More info:

My dissertation (ptbr): http://www.teses.usp.br/teses/disponiveis/55/55134/tde-05022018-090740/pt-br.php
EACL paper: http://www.aclweb.org/anthology/E17-1030
STIL paper: http://aclweb.org/anthology/W17-6618

If you want to use my data, please send me a e-mail.



# How to use a trained model?

Once you trained and saved a model, you can use a special class for loading each task. First, you need to import them:

```python
from deepbond.task import SentenceBoundaryDetector, FillerDetector, EditDisfDetector
```

Then you need to specify the prediciton model for lexical data and the prediction model for prosodic data:

```python
sbd = SentenceBoundaryDetector(l_model='rcnn', p_model='none', verbose=True)
```

In the third step you need to set the trained model name (which should be stored in `data/models/`):
```python
sbd.set_model_id('SS_TEXT_CINDERELA')
```

Finally, you can get your predictions by passing your texts to .detect():
```python
preds = sbd.detect(texts=your_texts)
```

See `example_test.py` for more details.


# How can I apply the entire pipeline?

```python
from deepbond import Pipeline # import pipeline
```

You need to instatiate each independent model for Sentence Segmentation and Disfluency Detection. Something like this:
```python
sbd = SentenceBoundaryDetector(l_model='rcnn', p_model='none', verbose=True)
sbd.set_model_id('SS_TEXT_CINDERELA')

fd = FillerDetector(l_model='rcnn', p_model='none', verbose=True)
fd.set_model_id('FILLERS_TEXT_CINDERELA')
fd.restrict_wordset()

edd = EditDisfDetector(l_model='rcnn', p_model='none', verbose=True)
edd.set_model_id('EDITDISFS_TEXT_CINDERELA')
```

Then you can pass them as parameters to the Pipeline's constructor and call .fit() method on your text :
```python
p = Pipeline(sbd, fd, edd, verbose=False)
preds = p.fit(texts=your_texts)
```

See `example_pipeline.py` for more details. 



# How can I train my own model?

## load data

You can use prepared trainig/testing sets specified in `loader.py` by setting the --'dataset option, but you are also able to specify a --'dataset-dir where each file consists of a training sample.

## extra resources

If you would like to use an embedding model, you should specify the type using --'embedding-type and the binary model to --'embedding-file. And if you would like to use an part-of-speech tagger, you should set the --'without-pos flag to false and specify --'pos-type and --'pos-file to a PoS Tagger model.

## building a new model

First, the imports:
```python
from deepbond.train import configure, train, get_default_options
```

Then setting the name and the save directory for your model:
```python
options['id'] = 'SS-EXAMPLE-CINDERELA'
options['model_dir'] = 'data/models/'+options['id']+'/' 	# dir where the model will be saved (data/models/:id:/)
```

Set the task:
```python
options['task'] = 'ss' 							# options are ss/dd_fillers/dd_editdisfs
```

Set the dataset or dataset-dir:
```python
# options['dataset'] = 'controle' 				# see loader.py (used only for training and error analysis)
options['dataset_dir'] = 'data/corpus/SS/Cinderela/' 	# only DA data will be used
```

Set the embeddings and PoS options:
```python
options['without_embeddings'] = False
options['emb_type'] = 'word2vec'				# method used for generate embeddings: see features/embeddings
options['emb_file'] = 'path/to/emb_model.bin'

options['without_pos'] = True
options['pos_type'] = 'nlpnet'					
options['pos_file'] = 'path/to/postagger/'		
```

Finally, call configure() and deepbond will create the directiries for your id and you can fit and save your model by calling train()
```python
configure(options)
train(options)
```

See `example_train.py` or `example_train_without_setup` for more details.


# Dependencies

See `requirements.txt` (you can easily install them with `pip3 install requirements.txt`)
