# deepbondd
Deep neural approach to Boundary and Disfluency Detection

This is part of my MSc project. More info:
\
[My dissertation (ptbr)](http://www.teses.usp.br/teses/disponiveis/55/55134/tde-05022018-090740/pt-br.php)
 • 
[EACL paper](http://www.aclweb.org/anthology/E17-1030)
 • 
[STIL paper](http://aclweb.org/anthology/W17-6618)
 • 
[PROPOR paper](https://www.researchgate.net/publication/327223308_Sentence_Segmentation_and_Disfluency_Detection_in_Narrative_Transcripts_from_Neuropsychological_Tests_13th_International_Conference_PROPOR_2018_Canela_Brazil_September_24-26_2018_Proceedings)

If you want to use my data, please send me an e-mail. A step-by-step tutorial (in portuguese) can be [accessed here](https://mtreviso.github.io/deepbond/tutorial.html).


# Dependencies and Installation

Optional: It is a good idea to create a virtualenv before installing the dependencies.

You can easily install the dependencies with: 
```bash
pip3 install -r requirements.txt
```

And then run the install command:
```bash
python3 setup.py install
```

Make sure both `pip` and `python` are bound to version 3.


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
from deepbond import Pipeline
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
options['model_dir'] = 'data/models/'+options['id']+'/'  # dir where the model will be saved (data/models/:id:/)
```

Set the task:
```python
options['task'] = 'ss'  # options are ss/dd_fillers/dd_editdisfs
```

Set the dataset or dataset-dir:
```python
# options['dataset'] = 'controle'  # see loader.py (used only for training and error analysis)
options['dataset_dir'] = 'data/corpus/SS/Cinderela/'  # only DA data will be used
```

Set the embeddings and PoS options:
```python
options['without_embeddings'] = False
options['emb_type'] = 'word2vec'  # method used for generate embeddings: see features/embeddings
options['emb_file'] = 'path/to/emb_model.bin'

options['without_pos'] = False
options['pos_type'] = 'nlpnet'					
options['pos_file'] = 'path/to/postagger/'		
```

Finally, call configure() and deepbond will create the directiries for your id and you can fit and save your model by calling train()
```python
configure(options)
train(options)
```

See `example_train.py` or `example_train_without_setup.sh` for more details.


# Arguments quick reference table

<table class="rich-diff-level-one"> <thead> <tr>
<th width="28%">Option</th>
<th width="14%">Default</th>
<th>Description</th>
</tr> </thead> <tbody>
<tr>
<td>
<code>-h</code> <code>--help</code>
</td>
<td></td>
<td>Show this help message and exit</td>
</tr>


<!-- general -->
<tr>
<td>
<code>--id</code>
</td>
<td></td>
<td>Name of configuration file in <code>configs/</code> folder (required)</td>
</tr>

<tr>
<td><code>--task</code></td>
<td><code>ss</code></td>
<td>Select the desired task:
<br><code>ss</code> for sentence boundaries 
<br><code>dd_fillers</code> for fillers
<br><code>dd_editdisfs</code> for edit disfluencies</td>
</tr>

<tr>
<td><code>--models</code></td>
<td><code>rcnn rcnn</code></td>
<td>A model for lexical info and another model for prosodic info. Options are:
<code>rcnn, rcnn_crf, cnn, rnn, mlp, crf, none</code>. Set <code>none</code> for not use a model
</tr>



<!-- io -->
<tr>
<td><code>-l</code> <code>--load</code></td>
<td><code>None</code></td>
<td>Load a trained model for the specified <code>id</code></td>
</tr>

<tr>
<td><code>-s</code> <code>--save</code></td>
<td><code>None</code></td>
<td>Save the trained model for the specified <code>id</code></td>
</tr>

<tr>
<td><code>--gpu</code></td>
<td></td>
<td>Run on GPU instead of on CPU</td>
</tr>

<tr>
<td><code>--model-dir</code></td>
<td><code>data/models/:id:</code></td>
<td>Directory to save/load data, model, log, etc.</td>
</tr>

<tr>
<td><code>--save-predictions</code></td>
<td></td>
<td>Path to save train and validation predictions in <code>data/saves/</code></td>
</tr>



<!-- hyperparams -->
<tr>
<td><code>--window-size</code></td>
<td><code>7</code></td>
<td>Size of the sliding window</td>
</tr>
<tr>
<td><code>--split-ratio</code></td>
<td><code>0.6</code></td>
<td>Ratio [0,1] to split the dataset into train/test</td>
</tr>
<tr>
<td><code>--train-strategy</code></td>
<td><code>bucket</code></td>
<td>Strategy for training: <code>bucket, window, padding, dicted</code></td>
</tr>
<tr>
<td><code>--epochs</code></td>
<td><code>20</code></td>
<td>Number of epochs</td>
</tr>
<tr>
<td><code>--batch-size</code></td>
<td><code>32</code></td>
<td>Size of data batches</td>
</tr>
<tr>
<td><code>--kfold</code></td>
<td><code>5</code></td>
<td>Number of folds to evaluate the model</td>
</tr>
<tr>
<td><code>--val-split</code></td>
<td><code>0.0</code></td>
<td>Ratio [0,1] to split the train dataset into train/validation (if 0 then alpha will be calculated using training data)</td>
</tr>

<tr>
<td><code>--dataset</code></td>
<td></td>
<td>One of: <code>constituicao, constituicao_mini, pucrs_usp, pucrs_constituicao, controle, ccl, da</code><br></td>
</tr>

<tr>
<td><code>--dataset-dir</code></td>
<td></td>
<td>Path to a corpus directory where each file is a sample</td>
</tr>


<tr>
<td><code>--extra-data</code></td>
<td></td>
<td>Add extra dataset as extension for training</td>
</tr>


<!-- pos -->
<tr>
<td><code>--pos-type</code></td>
<td><code>nlpnet</code></td>
<td>Tagger used POS features: nlpnet or pickled tagger</td>
</tr>
<tr>
<td><code>--pos-file</code></td>
<td><code>data/resource/pos-pt/</code></td>
<td>Path to pos tagger (can be a pickled object) resources</td>
</tr>
<tr>
<td><code>--without-pos</code></td>
<td></td>
<td>Do not use POS features</td>
</tr>


<!-- embeddings -->
<tr>
<td><code>--emb-type</code></td>
<td><code>word2vec</code></td>
<td>Method used to induce embeddings: word2vec/wang2vec/fasttext/glove/fonseca</td>
</tr>
<tr>
<td><code>--emb-file</code></td>
<td></td>
<td>Path to a binary embedding model file</td>
</tr>
<tr>
<td><code>--without-emb</code></td>
<td></td>
<td>Do not use embeddings</td>
</tr>


<!-- handcrafted -->
<tr>
<td><code>--use-handcrafted</code></td>
<td></td>
<td>Use handcrafted features (useful for detecting edit disfluencies)</td>
</tr>


<!-- prosodic -->
<tr>
<td><code>--prosodic-type</code></td>
<td><code>principal</code></td>
<td>Method used for select phones of a word: <code>principal</code> or <code>padding</code></td>
</tr>
<tr>
<td><code>--prosodic-classify</code></td>
<td></td>
<td>Classify prosodic info according to consonants</td>
</tr>
</tbody>
</table>



# Cite

If you use DeepBonDD, please cite this paper:

```
@inproceedings{treviso2018sentence,
  author = "Marcos Vinícius Treviso and Sandra Maria Aluísio",
  title = "Sentence Segmentation and Disfluency Detection in Narrative Transcripts from Neuropsychological Tests",
  booktitle = "Computational Processing of the Portuguese Language (PROPOR)",
  year = "2018",
  publisher = "Springer International Publishing",
  pages = "409--418",
}
```

# License
MIT. See the [LICENSE](LICENSE) file for more details.


