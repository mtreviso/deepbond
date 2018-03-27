import logging
from deepbond.features import Features
from deepbond.dataset import *

import deepbond.models.lexical as LexicalModels
import deepbond.models.prosodic as ProsodicModels
import deepbond.models.strategies as Strategies

logger = logging.getLogger(__name__)


def get_path_from_dataset(dataset):
	l, p = None, None
	paths = {

		# CINDERELA DATA:
		'controle': {
			'lexical': 'data/corpus/SS/Controle/',
			'prosodic': 'data/prosodic/controle.csv'
		},
		'ccl': {
			'lexical': 'data/corpus/SS/CCL-A/',
			'prosodic': 'data/prosodic/ccl.csv'
		},
		'da': {
			'lexical': 'data/corpus/SS/DA-Leve/',
			'prosodic': None
		},

		# FILLERS:
		'controle_fillers': {
			'lexical': 'data/corpus/Fillers/Controle/',
			'prosodic': 'data/prosodic/controle.csv'
		},
		'ccl_fillers': {
			'lexical': 'data/corpus/Fillers/CCL-A/',
			'prosodic': 'data/prosodic/ccl.csv'
		},
		'da_fillers': {
			'lexical': 'data/corpus/Fillers/DA-Leve/',
			'prosodic': None
		},

		# CORRECAO DO É:
		'controle_fillers_eh': {
			'lexical': 'data/corpus/Fillers-Eh/Controle/',
			'prosodic': 'data/prosodic/controle.csv'
		},
		'ccl_fillers_eh': {
			'lexical': 'data/corpus/Fillers-Eh/CCL-A/',
			'prosodic': 'data/prosodic/ccl.csv'
		},
		'da_fillers_eh': {
			'lexical': 'data/corpus/Fillers-Eh/DA-Leve/',
			'prosodic': None
		},

		# EDIT DISFS:
		'controle_editdisfs': {
			'lexical': 'data/corpus/EditDisfs-Specific/Controle/',
			'prosodic': 'data/prosodic/controle.csv'
		},
		'ccl_editdisfs': {
			'lexical': 'data/corpus/EditDisfs-Specific/CCL-A/',
			'prosodic': 'data/prosodic/ccl.csv'
		},
		'da_editdisfs': {
			'lexical': 'data/corpus/EditDisfs-Specific/DA-Leve/',
			'prosodic': None
		},

		# EDIT DISFS BINARY:
		'controle_editdisfs_binary': {
			'lexical': 'data/corpus/EditDisfs-Binary/Controle/',
			'prosodic': 'data/prosodic/controle.csv'
		},
		'ccl_editdisfs_binary': {
			'lexical': 'data/corpus/EditDisfs-Binary/CCL-A/',
			'prosodic': 'data/prosodic/ccl.csv'
		},
		'da_editdisfs_binary': {
			'lexical': 'data/corpus/EditDisfs-Binary/DA-Leve/',
			'prosodic': None
		},

		# EDIT DISFS WITHOUT FILLERS:
		'controle_editdisfs_wo_fillers': {
			'lexical': 'data/corpus/EditDisfs-Specific-wo-Fillers/Controle/',
			'prosodic': 'data/prosodic/controle.csv'
		},
		'ccl_editdisfs_wo_fillers': {
			'lexical': 'data/corpus/EditDisfs-Specific-wo-Fillers/CCL-A/',
			'prosodic': 'data/prosodic/ccl.csv'
		},
		'da_editdisfs_wo_fillers': {
			'lexical': 'data/corpus/EditDisfs-Specific-wo-Fillers/DA-Leve/',
			'prosodic': None
		},

		# EDIT DISFS WITHOUT FILLERS_EH:
		'controle_editdisfs_wo_fillers_eh': {
			'lexical': 'data/corpus/EditDisfs-Specific-wo-Fillers-Eh/Controle/',
			'prosodic': 'data/prosodic/controle.csv'
		},
		'ccl_editdisfs_wo_fillers_eh': {
			'lexical': 'data/corpus/EditDisfs-Specific-wo-Fillers-Eh/CCL-A/',
			'prosodic': 'data/prosodic/ccl.csv'
		},
		'da_editdisfs_wo_fillers_eh': {
			'lexical': 'data/corpus/EditDisfs-Specific-wo-Fillers-Eh/DA-Leve/',
			'prosodic': None
		},

		# EDIT DISFS BINARY WITHOUT FILLERS:
		'controle_editdisfs_wo_fillers_binary': {
			'lexical': 'data/corpus/EditDisfs-Binary-wo-Fillers/Controle/',
			'prosodic': 'data/prosodic/controle.csv'
		},
		'ccl_editdisfs_wo_fillers_binary': {
			'lexical': 'data/corpus/EditDisfs-Binary-wo-Fillers/CCL-A/',
			'prosodic': 'data/prosodic/ccl.csv'
		},
		'da_editdisfs_wo_fillers_binary': {
			'lexical': 'data/corpus/EditDisfs-Binary-wo-Fillers/DA-Leve/',
			'prosodic': None
		},

		# PIPELINE:
		'controle_pipeline': {
			'lexical': 'data/pipeline/3-editdisfs/data/Controle/',
			'prosodic': 'data/prosodic/controle.csv'
		},
		'ccl_pipeline': {
			'lexical': 'data/pipeline/3-editdisfs/data/CCL-A/',
			'prosodic': 'data/prosodic/ccl.csv'
		},
		'da_pipeline': {
			'lexical': 'data/pipeline/3-editdisfs/data/DA-Leve/',
			'prosodic': None
		},


		# ABCD CORPUS:
		'abcd_controle': {
			'lexical': 'data/corpus/ABCD/NLS_out/',
			'prosodic': None
		},
		'abcd_ccl': {
			'lexical': 'data/corpus/ABCD/CCL_out/',
			'prosodic': None
		},

		# CONSTITUICAO:
		'constituicao': {
			'lexical': 'data/corpus/Constituicao-Filtrada-Artigos/',
			'prosodic': 'data/prosodic/constituicao_filtrada_artigos.csv'
		},
		'constituicao_mini': {
			'lexical': 'data/corpus/Constituicao-Mini-Filtrada-Artigos/',
			'prosodic': 'data/prosodic/constituicao_mini_filtrada_artigos_ok.csv'
		},

		# PUCRS:
		'pucrs_controle': {
			'lexical': 'data/corpus/PUCRS/Controle/',
			'prosodic': 'data/prosodic/pucrs_controle.csv'
		},
		'pucrs_ccl': {
			'lexical': 'data/corpus/PUCRS/CCL-A/',
			'prosodic': 'data/prosodic/pucrs_ccl.csv'
		},

		# EXTRA DATA:
		'psfl': {
			'lexical': 'data/corpus/Extra/ParaSeuFilhoLer/',
			'prosodic': None
		},
		'folinha': {
			'lexical': 'data/corpus/Extra/Folinha/',
			'prosodic': None
		},
		'mundo_estranho': {
			'lexical': 'data/corpus/Extra/MundoEstranho/',
			'prosodic': None
		},
		'chc': {
			'lexical': 'data/corpus/Extra/CHC/',
			'prosodic': None
		},
		'livros_didativos': {
			'lexical': 'data/corpus/Extra/LivrosDidativos/',
			'prosodic': None
		},

		# PUCRS TEST:
		'pucrs_pont': {
			'lexical': 'data/corpus/PUCRS/Pont/',
			'prosodic': None
		},

		# CACHORRO DATA FROM PUCRS:
		'cachorro':{
			'lexical': 'data/corpus/Cachorro/',
			'prosodic': None
		},
		'cachorro_ccl': {
			'lexical': 'data/corpus/Cachorro/CCL-Baixa-Escolaridade/',
			'prosodic': None
		},
		'cachorro_da_analfabetos': {
			'lexical': 'data/corpus/Cachorro/DA-Analfabetos/',
			'prosodic': None
		},
		'cachorro_da_baixa': {
			'lexical': 'data/corpus/Cachorro/DA-Baixa-Escolaridade/',
			'prosodic': None
		},
		'cachorro_saudaveis_alta': {
			'lexical': 'data/corpus/Cachorro/Saudaveis-Alta-Escolaridade/',
			'prosodic': None
		},
		'cachorro_saudaveis_analfabetos': {
			'lexical': 'data/corpus/Cachorro/Saudaveis-Analfabetos/',
			'prosodic': None
		},
		'cachorro_saudaveis_baixa': {
			'lexical': 'data/corpus/Cachorro/Saudaveis-Baixa-Escolaridade/',
			'prosodic': None
		},
		'cachorro_da_all': {
			'lexical': 'data/corpus/Cachorro/DA-All/',
			'prosodic': None
		},
		'cachorro_saudaveis_all': {
			'lexical': 'data/corpus/Cachorro/Saudaveis-All/',
			'prosodic': None
		},
		'cachorro_all': {
			'lexical': 'data/corpus/Cachorro/All/',
			'prosodic': None
		}
	}
	if 'pucrs_usp' in dataset:
		dataset = 'pucrs_'+dataset[len('pucrs_usp_'):]
	if dataset in ['constituicao_controle', 'constituicao_ccl']:
		dataset = dataset[len('constituicao_'):]
	return {'text_dir': paths[dataset]['lexical'], 'prosodic_file': paths[dataset]['prosodic']}


def load_dataset(dataset, extra=False, vocabulary=None, task='ss'):

	originals = []	# datasets with prosody >> used both for train and test
	extensions = []	# datasets that may not have prosody >> used only for train

	if dataset == 'constituicao':
		logger.info('Reading ConstituicaoDataSet...')
		ds_constituicao = ConstituicaoDataSet(**get_path_from_dataset('constituicao'), vocabulary=vocabulary)
		ds_constituicao.info()
		originals.append(ds_constituicao)
	elif dataset == 'constituicao_mini':
		logger.info('Reading ConstituicaoDataSet...')
		ds_constituicao = ConstituicaoDataSet(**get_path_from_dataset('constituicao_mini'), vocabulary=vocabulary)
		ds_constituicao.info()
		originals.append(ds_constituicao)
	
	elif dataset == 'constituicao_controle':
		logger.info('Reading ConstituicaoDataSet...')
		ds_constituicao = ConstituicaoDataSet(**get_path_from_dataset('constituicao'), vocabulary=vocabulary)
		ds_constituicao.info()
		extensions.append(ds_constituicao)
		logger.info('Reading DementiaDataSet...')
		ds_dementia_controle = DementiaDataSet(**get_path_from_dataset('controle'), vocabulary=vocabulary)
		ds_dementia_controle.info()
		originals.append(ds_dementia_controle)
	elif dataset == 'constituicao_ccl':
		logger.info('Reading ConstituicaoDataSet...')
		ds_constituicao = ConstituicaoDataSet(**get_path_from_dataset('constituicao'), vocabulary=vocabulary)
		ds_constituicao.info()
		extensions.append(ds_constituicao)
		logger.info('Reading DementiaDataSet...')
		ds_dementia_ccl = DementiaDataSet(**get_path_from_dataset('ccl'), vocabulary=vocabulary)
		ds_dementia_ccl.info()
		originals.append(ds_dementia_ccl)

	elif dataset == 'constituicao_mini_controle':
		logger.info('Reading ConstituicaoDataSet...')
		ds_constituicao = ConstituicaoDataSet(**get_path_from_dataset('constituicao_mini'), vocabulary=vocabulary)
		ds_constituicao.info()
		extensions.append(ds_constituicao)
		logger.info('Reading DementiaDataSet...')
		ds_dementia_controle = DementiaDataSet(**get_path_from_dataset('controle'), vocabulary=vocabulary)
		ds_dementia_controle.info()
		originals.append(ds_dementia_controle)
	elif dataset == 'constituicao_mini_ccl':
		logger.info('Reading ConstituicaoDataSet...')
		ds_constituicao = ConstituicaoDataSet(**get_path_from_dataset('constituicao_mini'), vocabulary=vocabulary)
		ds_constituicao.info()
		extensions.append(ds_constituicao)
		logger.info('Reading DementiaDataSet...')
		ds_dementia_ccl = DementiaDataSet(**get_path_from_dataset('ccl'), vocabulary=vocabulary)
		ds_dementia_ccl.info()
		originals.append(ds_dementia_ccl)

	elif 'pucrs_usp' in dataset or 'pucrs_constituicao' in dataset:
		logger.info('Reading DementiaPUCDataSet...')
		if 'controle' in dataset:
			ds_pucrs_controle = DementiaPUCDataSet(**get_path_from_dataset('pucrs_controle'), vocabulary=vocabulary)
			ds_pucrs_controle.info()
			originals.append(ds_pucrs_controle)
		if 'ccl' in dataset:
			ds_pucrs_ccl = DementiaPUCDataSet(**get_path_from_dataset('pucrs_ccl'), vocabulary=vocabulary)
			ds_pucrs_ccl.info()
			originals.append(ds_pucrs_ccl)
		if 'pont' in dataset:
			ds_pucrs_pont = DementiaPUCDataSet(**get_path_from_dataset('pucrs_pont'), vocabulary=vocabulary)
			ds_pucrs_pont.info()
			originals.append(ds_pucrs_pont)

			# add outros da pucrs para treino
			ds_pucrs_controle = DementiaPUCDataSet(**get_path_from_dataset('pucrs_controle'), vocabulary=vocabulary)
			ds_pucrs_controle.info()
			extensions.append(ds_pucrs_controle)
			ds_pucrs_ccl = DementiaPUCDataSet(**get_path_from_dataset('pucrs_ccl'), vocabulary=vocabulary)
			ds_pucrs_ccl.info()
			extensions.append(ds_pucrs_ccl)
		if 'pucrs_usp' in dataset:
			ds_dementia_controle = DementiaDataSet(**get_path_from_dataset('controle'), vocabulary=vocabulary)
			ds_dementia_ccl = DementiaDataSet(**get_path_from_dataset('ccl'), vocabulary=vocabulary)
			ds_dementia_da = DementiaDataSet(**get_path_from_dataset('da'), vocabulary=vocabulary)
			ds_dementia_controle.info()
			ds_dementia_ccl.info()
			ds_dementia_da.info()
			extensions.append(ds_dementia_controle)
			extensions.append(ds_dementia_ccl)
			extensions.append(ds_dementia_da)
		else:
			ds_constituicao = ConstituicaoDataSet(**get_path_from_dataset('constituicao'), vocabulary=vocabulary)
			ds_constituicao.info()
			extensions.append(ds_constituicao)
	
	elif dataset in ['controle', 'ccl', 'da']:
		logger.info('Reading DementiaDataSet...')
		ds_dementia_controle = DementiaDataSet(**get_path_from_dataset('controle'), vocabulary=vocabulary)
		ds_dementia_ccl = DementiaDataSet(**get_path_from_dataset('ccl'), vocabulary=vocabulary)
		ds_dementia_da = DementiaDataSet(**get_path_from_dataset('da'), vocabulary=vocabulary)
		ds_dementia_controle.info()
		ds_dementia_ccl.info()
		ds_dementia_da.info()
		for dname, dobj in zip(['controle', 'ccl', 'da'], [ds_dementia_controle, ds_dementia_ccl, ds_dementia_da]):
			if dname == dataset:
				originals.append(dobj)
			else:
				extensions.append(dobj)
	elif dataset in ['controle_fillers', 'ccl_fillers', 'da_fillers']:
		logger.info('Reading DementiaDataSet...')
		ds_dementia_controle = DementiaDataSet(**get_path_from_dataset('controle_fillers'), vocabulary=vocabulary)
		ds_dementia_ccl = DementiaDataSet(**get_path_from_dataset('ccl_fillers'), vocabulary=vocabulary)
		ds_dementia_da = DementiaDataSet(**get_path_from_dataset('da_fillers'), vocabulary=vocabulary)
		ds_dementia_controle.info()
		ds_dementia_ccl.info()
		ds_dementia_da.info()
		for dname, dobj in zip(['controle_fillers', 'ccl_fillers', 'da_fillers'], [ds_dementia_controle, ds_dementia_ccl, ds_dementia_da]):
			if dname == dataset:
				originals.append(dobj)
			else:
				extensions.append(dobj)

	elif dataset in ['controle_fillers_eh', 'ccl_fillers_eh', 'da_fillers_eh']:
		logger.info('Reading DementiaDataSet...')
		ds_dementia_controle = DementiaDataSet(**get_path_from_dataset('controle_fillers_eh'), vocabulary=vocabulary)
		ds_dementia_ccl = DementiaDataSet(**get_path_from_dataset('ccl_fillers_eh'), vocabulary=vocabulary)
		ds_dementia_da = DementiaDataSet(**get_path_from_dataset('da_fillers_eh'), vocabulary=vocabulary)
		ds_dementia_controle.info()
		ds_dementia_ccl.info()
		ds_dementia_da.info()
		for dname, dobj in zip(['controle_fillers_eh', 'ccl_fillers_eh', 'da_fillers_eh'], [ds_dementia_controle, ds_dementia_ccl, ds_dementia_da]):
			if dname == dataset:
				originals.append(dobj)
			else:
				extensions.append(dobj)
	
	elif dataset in ['controle_editdisfs', 'ccl_editdisfs', 'da_editdisfs']:
		logger.info('Reading DementiaDataSet...')
		ds_dementia_controle = DementiaDataSet(**get_path_from_dataset('controle_editdisfs'), vocabulary=vocabulary)
		ds_dementia_ccl = DementiaDataSet(**get_path_from_dataset('ccl_editdisfs'), vocabulary=vocabulary)
		ds_dementia_da = DementiaDataSet(**get_path_from_dataset('da_editdisfs'), vocabulary=vocabulary)
		ds_dementia_controle.info()
		ds_dementia_ccl.info()
		ds_dementia_da.info()
		for dname, dobj in zip(['controle_editdisfs', 'ccl_editdisfs', 'da_editdisfs'], [ds_dementia_controle, ds_dementia_ccl, ds_dementia_da]):
			if dname == dataset:
				originals.append(dobj)
			else:
				extensions.append(dobj)
	
	elif dataset in ['controle_editdisfs_wo_fillers', 'ccl_editdisfs_wo_fillers', 'da_editdisfs_wo_fillers']:
		logger.info('Reading DementiaDataSet...')
		ds_dementia_controle = DementiaDataSet(**get_path_from_dataset('controle_editdisfs_wo_fillers'), vocabulary=vocabulary)
		ds_dementia_ccl = DementiaDataSet(**get_path_from_dataset('ccl_editdisfs_wo_fillers'), vocabulary=vocabulary)
		ds_dementia_da = DementiaDataSet(**get_path_from_dataset('da_editdisfs_wo_fillers'), vocabulary=vocabulary)
		ds_dementia_controle.info()
		ds_dementia_ccl.info()
		ds_dementia_da.info()
		for dname, dobj in zip(['controle_editdisfs_wo_fillers', 'ccl_editdisfs_wo_fillers', 'da_editdisfs_wo_fillers'], [ds_dementia_controle, ds_dementia_ccl, ds_dementia_da]):
			if dname == dataset:
				originals.append(dobj)
			else:
				extensions.append(dobj)

	elif dataset in ['controle_editdisfs_wo_fillers_eh', 'ccl_editdisfs_wo_fillers_eh', 'da_editdisfs_wo_fillers_eh']:
		logger.info('Reading DementiaDataSet...')
		ds_dementia_controle = DementiaDataSet(**get_path_from_dataset('controle_editdisfs_wo_fillers_eh'), vocabulary=vocabulary)
		ds_dementia_ccl = DementiaDataSet(**get_path_from_dataset('ccl_editdisfs_wo_fillers_eh'), vocabulary=vocabulary)
		ds_dementia_da = DementiaDataSet(**get_path_from_dataset('da_editdisfs_wo_fillers_eh'), vocabulary=vocabulary)
		ds_dementia_controle.info()
		ds_dementia_ccl.info()
		ds_dementia_da.info()
		for dname, dobj in zip(['controle_editdisfs_wo_fillers_eh', 'ccl_editdisfs_wo_fillers_eh', 'da_editdisfs_wo_fillers_eh'], [ds_dementia_controle, ds_dementia_ccl, ds_dementia_da]):
			if dname == dataset:
				originals.append(dobj)
			else:
				extensions.append(dobj)

	elif dataset in ['controle_editdisfs_binary', 'ccl_editdisfs_binary', 'da_editdisfs_binary']:
		logger.info('Reading DementiaDataSet...')
		ds_dementia_controle = DementiaDataSet(**get_path_from_dataset('controle_editdisfs_binary'), vocabulary=vocabulary)
		ds_dementia_ccl = DementiaDataSet(**get_path_from_dataset('ccl_editdisfs_binary'), vocabulary=vocabulary)
		ds_dementia_da = DementiaDataSet(**get_path_from_dataset('da_editdisfs_binary'), vocabulary=vocabulary)
		ds_dementia_controle.info()
		ds_dementia_ccl.info()
		ds_dementia_da.info()
		for dname, dobj in zip(['controle_editdisfs_binary', 'ccl_editdisfs_binary', 'da_editdisfs_binary'], [ds_dementia_controle, ds_dementia_ccl, ds_dementia_da]):
			if dname == dataset:
				originals.append(dobj)
			else:
				extensions.append(dobj)
	
	elif dataset in ['controle_editdisfs_wo_fillers_binary', 'ccl_editdisfs_wo_fillers_binary', 'da_editdisfs_wo_fillers_binary']:
		logger.info('Reading DementiaDataSet...')
		ds_dementia_controle = DementiaDataSet(**get_path_from_dataset('controle_editdisfs_wo_fillers_binary'), vocabulary=vocabulary)
		ds_dementia_ccl = DementiaDataSet(**get_path_from_dataset('ccl_editdisfs_wo_fillers_binary'), vocabulary=vocabulary)
		ds_dementia_da = DementiaDataSet(**get_path_from_dataset('da_editdisfs_wo_fillers_binary'), vocabulary=vocabulary)
		ds_dementia_controle.info()
		ds_dementia_ccl.info()
		ds_dementia_da.info()
		for dname, dobj in zip(['controle_editdisfs_wo_fillers_binary', 'ccl_editdisfs_wo_fillers_binary', 'da_editdisfs_wo_fillers_binary'], [ds_dementia_controle, ds_dementia_ccl, ds_dementia_da]):
			if dname == dataset:
				originals.append(dobj)
			else:
				extensions.append(dobj)

	elif dataset in ['controle_pipeline', 'ccl_pipeline', 'da_pipeline']:
		logger.info('Reading DementiaDataSet...')
		ds_dementia_controle = DementiaDataSet(**get_path_from_dataset('controle_pipeline'), vocabulary=vocabulary)
		ds_dementia_ccl = DementiaDataSet(**get_path_from_dataset('ccl_pipeline'), vocabulary=vocabulary)
		ds_dementia_da = DementiaDataSet(**get_path_from_dataset('da_pipeline'), vocabulary=vocabulary)
		ds_dementia_controle.info()
		ds_dementia_ccl.info()
		ds_dementia_da.info()
		for dname, dobj in zip(['controle_pipeline', 'ccl_pipeline', 'da_pipeline'], [ds_dementia_controle, ds_dementia_ccl, ds_dementia_da]):
			if dname == dataset:
				originals.append(dobj)
			else:
				extensions.append(dobj)

	elif 'abcd' in dataset:
		logger.info('Reading DementiaDataSet...')
		ds_dementia_controle = DementiaDataSet(**get_path_from_dataset('controle'), vocabulary=vocabulary)
		ds_dementia_ccl = DementiaDataSet(**get_path_from_dataset('ccl'), vocabulary=vocabulary)
		ds_dementia_controle.info()
		ds_dementia_ccl.info()
		extensions.append(ds_dementia_controle)
		extensions.append(ds_dementia_ccl)
		# ds_dementia_da = DementiaDataSet(**get_path_from_dataset('da'), vocabulary=vocabulary)
		# ds_dementia_da.info()
		# extensions.append(ds_dementia_da)
		ds_abcd = ABCDDataSet(**get_path_from_dataset(dataset), vocabulary=vocabulary)
		ds_abcd.info()
		originals.append(ds_abcd)
	
	elif 'cachorro' in dataset:
		ds_dementia_controle = DementiaDataSet(**get_path_from_dataset('controle'), vocabulary=vocabulary)
		ds_dementia_ccl = DementiaDataSet(**get_path_from_dataset('ccl'), vocabulary=vocabulary)
		ds_dementia_da = DementiaDataSet(**get_path_from_dataset('da'), vocabulary=vocabulary)
		ds_dementia_controle.info()
		ds_dementia_ccl.info()
		ds_dementia_da.info()
		extensions.append(ds_dementia_controle)
		extensions.append(ds_dementia_ccl)
		extensions.append(ds_dementia_da)

		cachorros_datasets = ['cachorro_ccl', 'cachorro_da_analfabetos', 
							'cachorro_da_baixa', 'cachorro_saudaveis_alta', 
							'cachorro_saudaveis_analfabetos', 'cachorro_saudaveis_baixa']
		dataset_subname = dataset[9:]

		if 'all' in dataset_subname:
			sub = dataset_subname.split('_')[0] if '_' in dataset_subname else dataset_subname
			for cname in cachorros_datasets:
				if cname.split('_')[1] != sub:
					ds_dementia_x = DementiaPUCDataSet(**get_path_from_dataset(cname), vocabulary=vocabulary)
					ds_dementia_x.info()
					extensions.append(ds_dementia_x)
			ds_dementia_x = DementiaPUCDataSet(**get_path_from_dataset(dataset), vocabulary=vocabulary)
			ds_dementia_x.info()
			originals.append(ds_dementia_x)
		else:
			for cname in cachorros_datasets:
				ds_dementia_x = DementiaPUCDataSet(**get_path_from_dataset(cname), vocabulary=vocabulary)
				ds_dementia_x.info()
				if dataset == cname:
					originals.append(ds_dementia_x)
				else:
					extensions.append(ds_dementia_x)


	if extra:
		logger.info('Reading ParaSeuFilhoLerDataSet...')
		ds_extra_filho = ParaSeuFilhoLerDataSet(**get_path_from_dataset('psfl'), vocabulary=vocabulary)
		logger.info('Reading FolinhaDataSet...')
		ds_extra_folinha = FolinhaDataSet(**get_path_from_dataset('folinha'), vocabulary=vocabulary)
		# logger.info('Reading MundoEstranhoDataSet...')
		# ds_extra_mundo = MundoEstranhoDataSet(**get_path_from_dataset('mundo_estranho'), vocabulary=vocabulary)
		# logger.info('Reading CHCDataSet...')
		# ds_extra_chc = CHCDataSet(**get_path_from_dataset('chc'), vocabulary=vocabulary)
		# logger.info('Reading LivrosDidaticosDataSet...')
		# ds_extra_livros = LivrosDidaticosDataSet(**get_path_from_dataset('livros_didativos'), vocabulary=vocabulary)
		ds_extra_filho.info()
		ds_extra_folinha.info()
		# ds_extra_mundo.info()
		# ds_extra_chc.info()
		# ds_extra_livros.info()
		extensions.append(ds_extra_filho)
		extensions.append(ds_extra_folinha)
		# extensions.append(ds_extra_mundo)
		# extensions.append(ds_extra_chc)
		# extensions.append(ds_extra_livros)
		

	dsm = DataSetManager(originals=originals, extensions=extensions, task=task)
	dsm.info()
	return dsm


def build_dataset_from_data(texts, audios, task='ss'):
	ds_raw = RawDataSet(texts, audios)
	dsm = DataSetManager(originals=[ds_raw], extensions=[], task=task)
	return dsm

def load_features(POS_type, POS_file, embedding_type, embedding_file, prosodic_type, prosodic_classify, use_pos, use_embeddings, use_handcrafted):
	ft = Features(POS_type=POS_type, POS_file=POS_file, 
				prosodic_type=prosodic_type, prosodic_classify=prosodic_classify, 
				embedding_type=embedding_type, embedding_file=embedding_file,
				use_pos=use_pos, use_embeddings=use_embeddings, use_handcrafted=use_handcrafted)
	ft.info()
	return ft


def load_strategy(train_strategy, window_size=7, max_sentence_size=None):
	strategy = train_strategy
	if strategy == 'bucket':
		return Strategies.BucketStrategy(name=strategy, input_length=None)
	elif strategy == 'padding':
		return Strategies.PaddingStrategy(name=strategy, input_length=max_sentence_size)
	elif strategy == 'window':
		return Strategies.WindowStrategy(name=strategy, input_length=window_size)
	elif strategy == 'dicted':
		return Strategies.DictedStrategy(name=strategy, input_length=None)
	raise Exception('The strategy chosen was not implemented!')


def _select_model(Models, model_params, x):
	if x == 'rcnn':
		model = Models.RCNN(**model_params)
		params = {'filter_length':7, 'rnn':'GRU'}
	elif x == 'rcnn_crf':
		model = Models.RCNN_CRF(**model_params)
		params = {'filter_length':7, 'rnn':'GRU'}
	elif x == 'cnn':
		model = Models.CNN(**model_params)
		params = {'filter_length':7}
	elif x == 'rnn':
		model = Models.RecNN(**model_params)
		params = {'rnn':'GRU'}
	elif x == 'mlp':
		model = Models.MLP(**model_params)
		params = {}
	elif x == 'crf':
		model = Models.CRF(**model_params)
		params = {}
	elif x == 'none' or x is None:
		model = None
		params = {}
	else:
		raise Exception('Model %s not implemented' % x)
	return model, params


def load_models(lexical, prosodic, features, vocabulary, nb_classes, strategy):
	
	l_model_params = {
		'features': 		features,
		'vocabulary': 		vocabulary,
		'nb_classes': 		nb_classes,
		'input_length': 	strategy.input_length,
		'use_embeddings': 	features.use_embeddings,
		'use_pos': 			features.use_pos,
		'use_handcrafted':	features.use_handcrafted
	}
	l_model, l_params = _select_model(LexicalModels, l_model_params, lexical)


	p_model_params = {
		'features': 	features,
		'vocabulary': 	vocabulary,
		'nb_classes': 	nb_classes,
		'input_length': strategy.input_length
	}
	p_model, p_params = _select_model(ProsodicModels, p_model_params, prosodic)

	return l_model, l_params, p_model, p_params

