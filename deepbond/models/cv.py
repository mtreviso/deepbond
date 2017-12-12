import logging
import numpy as np

from sklearn.model_selection import KFold
from deepbond.utils import unvectorize
from deepbond.models.utils import save_to_file

logger = logging.getLogger(__name__)

class CrossValidation:

	def __init__(self, data_set_manager, train_manager, options):
		self.dsm = data_set_manager
		self.train_manager = train_manager
		self.epochs = options.epochs
		self.folds = options.kfold
		self.val_split = options.val_split
		l_name = options.models[0]
		p_name = options.models[1]
		self.predictions_filename = '%s-%s-%s-%s-%s' % (l_name, p_name, options.dataset, options.pos_type, options.emb_type)
		self.predictions_dirname = options.save_predictions
		self.save_predictions = options.save_predictions
		self.task = options.task


	def run(self, verbose=True):
		'''
		Micro averaged or Macro averaged F1?
		http://stats.stackexchange.com/questions/156923/should-i-make-decisions-based-on-micro-averaged-or-macro-averaged-evaluation-mea
		paper sobre: http://www.sciencedirect.com/science/article/pii/S0306457309000259
		'''
		lexical_stats = [0, 0, 0]
		prosodic_stats = [0, 0, 0]
		combiner_stats = [0, 0, 0]
		best_partition = 0
		
		n_samples = sum(x.nb_texts for x in self.dsm.originals)
		if self.folds == -1 or n_samples < self.folds: # leave one out
			self.folds = n_samples
		kf = KFold(n_splits=self.folds, shuffle=True)
		placeholder = np.zeros(n_samples)

		for k, (train_index, test_index) in enumerate(kf.split(placeholder)):
			logger.info('K fold: {} of {}'.format(k+1, self.folds))
			
			logger.info('Train/Test summary: ')
			ds_train, ds_test = self.dsm.split_by_index(train_index, test_index)
			ds_train.info()
			ds_test.info()

			ls, ps, cs, best_p, = self.train_manager.train(ds_train, ds_test, nb_epoch=self.epochs, verbose=verbose)
			predictions = self.train_manager.get_test_predictions()

			lexical_stats 	= [x+y for x,y in zip(lexical_stats, ls)]
			prosodic_stats 	= [x+y for x,y in zip(prosodic_stats, ps)]
			combiner_stats	= [x+y for x,y in zip(combiner_stats, cs)]
			best_partition 	= best_partition + best_p

			if self.save_predictions:
				self._save_predictions(ds_test.word_texts, ds_test.shuffle_indexes, predictions, fold=k)

			logger.info('\n---\n')
		
		if verbose:
			self._show_stats(lexical_stats, prosodic_stats, combiner_stats, best_partition)



	def nested_run(self, verbose=True):
		'''
		Nested CV vs Non-Nested:
		http://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
		http://stats.stackexchange.com/questions/65128/nested-cross-validation-for-model-selection
		http://stats.stackexchange.com/questions/167066/double-nested-wrapper-crossvalidation-final-trained-model
		'''
		# hyperparam to be tuned
		best_p = []
		n_splits = 3
		n_samples = sum(x.nb_texts for x in self.dsm.originals)
		kf = KFold(n_splits=n_splits, shuffle=True)
		placeholder = np.zeros(n_samples)
		
		# Inner CV
		for k, (train_index, val_index) in enumerate(kf.split(placeholder)):
			logger.info('K fold: {}'.format(k+1))
			logger.info('Train/Test summary: ')
			ds_train, ds_val = self.dsm.split_by_index(train_index, val_index)
			ds_train.info()
			ds_val.info()
			self.train_manager.train(ds_train, ds_test=None, ds_val=ds_val, nb_epoch=10, verbose=verbose)
			best_p.append(self.train_manager.c_instance.best_p)
		
		logger.debug("Avg p that max F1: %.2f (%.2f)" % (np.mean(best_p), np.std(best_p)))

		# Outer CV
		self.train_manager.best_p = np.mean(best_p)
		self.run()
		
	def _save_predictions(self, original_texts, original_indexes, predictions, fold=0):
		fname = self.predictions_filename + '-fold_%d' % fold
		dname = self.predictions_dirname
		if isinstance(predictions[0][0], (list, np.ndarray)):
			predictions = list(map(unvectorize, predictions))
		save_to_file(original_texts, original_indexes, predictions, fname=fname, dname=dname, task=self.task)

	def _show_stats(self, lexical_stats, prosodic_stats, combiner_stats, best_partition):
		logger.debug("Text Precision  = %.4f  Recall = %.4f F-Measure = %.4f" % (lexical_stats[1]/self.folds, lexical_stats[2]/self.folds, lexical_stats[0]/self.folds))
		logger.debug("Audio Precision = %.4f  Recall = %.4f F-Measure = %.4f" % (prosodic_stats[1]/self.folds, prosodic_stats[2]/self.folds, prosodic_stats[0]/self.folds))
		logger.debug("All Precision   = %.4f  Recall = %.4f F-Measure = %.4f" % (combiner_stats[1]/self.folds, combiner_stats[2]/self.folds, combiner_stats[0]/self.folds))
		logger.debug('TO CSV: %.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.1f\t%.1f' % (lexical_stats[1]/self.folds, lexical_stats[2]/self.folds, lexical_stats[0]/self.folds,
																								  prosodic_stats[1]/self.folds, prosodic_stats[2]/self.folds, prosodic_stats[0]/self.folds,
																								  combiner_stats[1]/self.folds, combiner_stats[2]/self.folds, combiner_stats[0]/self.folds,
																								  best_partition/self.folds, 1-best_partition/self.folds))
