from deepbond.train import configure, train, get_default_options

options = get_default_options()
for x, y in options.items():
	print('%s: %s' % (x, y))

embpath='/media/treviso/FEJ/Embeddings-Deepbond/ptbr/word2vec/pt_word2vec_sg_600.emb'



options['id'] = 'SS-EXAMPLE-CINDERELA-DA'
options['model_dir'] = 'data/models/'+options['id']+'/' 	# dir where the model will be saved (data/models/:id:/)

options['task'] = 'ss' 							# one of ss/dd_fillers/dd_editdisfs/ssdd
# options['dataset'] = 'controle' 				# see loader.py (used only for training and error analysis)
options['dataset_dir'] = 'data/corpus/SS/DA-Leve/' 	# only DA data will be used

options['emb_type'] = 'word2vec'		# method used for generate embeddings: see features/embeddings
options['emb_file'] = embpath			# e.g. Embeddings-Deepbond/ptbr/word2vec/pt_word2vec_sg_600.emb

configure(options)
train(options)

