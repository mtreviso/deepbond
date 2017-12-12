import sys
import logging
import numpy as np
from random import shuffle as shuffle_list

from sklearn.utils import compute_class_weight

from deepbond.dataset import DataSet
from deepbond.statistics import Statistics
from deepbond.models.callbacks import ModelStats, EarlyStopping
from deepbond.utils import unvectorize, unroll, reshape_like, bucketize, reorder_buckets

logger = logging.getLogger(__name__)

class FillerModel:

	def __init__(self, model, strategy, batch_size=32):
		self.model = model
		self.strategy = strategy
		self.train_stats = None
		self.test_stats = None
		self.val_stats = None
		self.batch_size = batch_size

	def build(self, **params):
		self.model.build(**params)

	def train(self, ds_train, ds_test=None, ds_val=None, nb_epoch=20, verbose=True):
		logger.info('Training...')
		callbacks = []
		nb_train = len(ds_train.texts)
		val_tuple = None
		
		logger.info('Preparing train input...')
		train_data, train_y, train_gold = self.model.prepare_input(ds_train, self.strategy)
		self.train_stats = ModelStats(train_data, train_gold, cls=self, what='Train', verbose=verbose)
		callbacks.append(self.train_stats)

		logger.info('Calculating sample weight...')
		classes = list(range(self.model.nb_classes))
		outputs = np.array(unroll(train_gold))
		class_weight = dict(zip(classes, compute_class_weight('balanced', classes, outputs)))
		sample_weight = np.array(list(map(lambda x: list(map(class_weight.__getitem__, x)), train_gold)))
		if self.strategy.name == 'window':
			sample_weight = class_weight

		if ds_val is not None:
			logger.info('Preparing val input...')
			val_data, val_y, val_gold = self.model.prepare_input(ds_val, self.strategy)
			val_sw = np.array(list(map(lambda x: list(map(class_weight.__getitem__, x)), val_gold)))
			val_tuple = (val_data, val_y, val_sw)
			self.val_stats = ModelStats(val_data, val_gold, cls=self, what='Val', verbose=verbose)
			callbacks.append(self.val_stats)

		if ds_test is not None:
			logger.info('Preparing test input...')
			test_data, test_y, test_gold = self.model.prepare_input(ds_test, self.strategy)
			self.test_stats = ModelStats(test_data, test_gold, cls=self, what='Test', verbose=verbose)
			callbacks.append(self.test_stats)
		
		# callbacks.append(EarlyStopping(monitor='val_loss', patience=3, mode='auto'))

		logger.info('Fitting...')
		self._fit(train_data, train_y, val_data=val_tuple, sample_weight=sample_weight, 
				nb_epoch=nb_epoch, callbacks=callbacks)

		if self.strategy.name == 'dicted':
			tf, tp, tr, tpred, tgold = self.evaluate(train_data, train_gold, return_predictions=True, return_golds=True, verbose=verbose)
			self.train_stats.stats = [tf,tp,tr] # f1, prec, rec
			self.train_stats.best_pred = tpred
			self.train_stats.best_gold = tgold
			bf, bp, br, bpred, bgold = self.evaluate(test_data, test_gold, return_predictions=True, return_golds=True, verbose=verbose)
			self.test_stats.stats = [bf, bp, br] # f1, prec, rec
			self.test_stats.best_pred = bpred
			self.test_stats.best_gold = bgold


	def _fit(self, train_data, train_y, val_data=None, sample_weight=None, nb_epoch=20, callbacks=[]):
		if self.strategy.name == 'bucket':
			train_buckets 	= bucketize(train_y, max_bucket_size=self.batch_size)
			train_generator = self.model.get_batch(train_data, train_y, sample_weight, *train_buckets, shuffle=True)
			val_generator 	= None
			nb_val_samples	= None
			if val_data is not None:
				val_buckets 	= bucketize(val_data[1], max_bucket_size=self.batch_size)
				val_generator	= self.model.get_batch(*val_data, *val_buckets, shuffle=False)
				nb_val_samples	= len(val_data[1])
			self.model.classifier.fit_generator(train_generator, samples_per_epoch=len(train_buckets[0]),
												validation_data=val_generator, nb_val_samples=nb_val_samples,
												nb_epoch=nb_epoch, callbacks=callbacks)
		elif self.strategy.name == 'dicted':
			self.model.classifier.fit(train_data, train_y)			
		elif self.strategy.name == 'window':
			self.model.classifier.fit(train_data, train_y, nb_epoch=nb_epoch, batch_size=self.batch_size, 
									validation_data=val_data, callbacks=callbacks, class_weight=sample_weight)
		else:
			self.model.classifier.fit(train_data, train_y, nb_epoch=nb_epoch, batch_size=self.batch_size, 
									validation_data=val_data, callbacks=callbacks, sample_weight=sample_weight)


	def _predict_on_batch(self, generator, val_samples, return_gold=False, verbose=False):
		# !!!fix keras predict_generator function and make a pull request!!!
		# preds = self.model.classifier.predict_generator(generator, val_samples=val_samples)
		preds, golds = [], []
		if verbose:
			print('')
		for i, (X, Y, _) in enumerate(generator):
			if verbose:
				sys.stdout.write('Prediction %d/%d \r' % (i+1, val_samples))
				sys.stdout.flush()
			out = self.model.classifier.predict_on_batch(X)
			for y in out:
				preds.append(y)
			if return_gold:
				for bucket in Y.values():
					for y in bucket:
						golds.append(y)
		if return_gold:
			return preds, golds
		return preds

	def predict(self, data, verbose=False):
		if self.strategy.name == 'bucket':
			lengths, data_by_length = bucketize(data[0], max_bucket_size=self.batch_size)
			pred_generator = self.model.get_batch(data, data[0], None, lengths, data_by_length, shuffle=False, kind='predict')
			preds = self._predict_on_batch(pred_generator, len(data_by_length), verbose=verbose)
			preds = self.strategy.unprepare(preds)
			preds, _ = reorder_buckets(preds, preds, lengths, data_by_length)
			return preds, lengths, data_by_length
		elif self.strategy.name == 'dicted':
			labels = [str(l) for l in range(self.model.nb_classes)]
			flat_dict = lambda x: [[[w[l] for l in labels] for w in sent] for sent in x]
			preds = self.model.classifier.predict_marginals(data)
			preds = flat_dict(preds)
			preds = [np.array(p) for p in preds]
		else:
			preds = self.model.classifier.predict(data, batch_size=self.batch_size, verbose=verbose)
			preds = self.strategy.unprepare(preds)
		return preds

	def evaluate(self, data, gold, last_is_dot=True, return_predictions=False, return_golds=False, verbose=True):
		if self.strategy.name == 'bucket':
			lengths, data_by_length = bucketize(data[0], max_bucket_size=self.batch_size)
			pred_generator 			= self.model.get_batch(data, gold, None, lengths, data_by_length, shuffle=False, kind='predict')
			preds_vec, golds		= self._predict_on_batch(pred_generator, len(data_by_length), return_gold=True, verbose=verbose)
			preds_vec, golds 		= reorder_buckets(preds_vec, golds, lengths, data_by_length)
			preds_vec 				= self.strategy.unprepare(preds_vec)
			preds 					= [unvectorize(np.array(p)) for p in preds_vec] 
		elif self.strategy.name == 'dicted':
			golds = gold
			preds_vec = self.predict(data, verbose=verbose)
			preds = [unvectorize(p) for p in preds_vec]
		elif self.strategy.name == 'window':
			golds = gold
			preds_vec = self.predict(data, verbose=verbose)
			preds = unvectorize(np.array(preds_vec))
			preds = reshape_like(preds, map_with=golds)
			preds_vec = reshape_like(preds_vec, map_with=golds)
		else:
			golds = gold
			preds_vec = self.predict(data, verbose=verbose)
			preds = unvectorize(np.array(preds_vec))
		stats = list(Statistics.get_metrics(unroll(golds), unroll(preds)))
		plus = []
		if return_predictions:
			plus.append(preds_vec)
		if return_golds:
			plus.append(golds)
		return stats + plus

	def save_weights(self, filename):
		self.model.save_weights(filename)

	def load_weights(self, filename):
		self.model.load_weights(filename)


################################################################### 
################################################################### 
################################################################### 
################################################################### 
################################################################### 
################################################################### 
################################################################### 
################################################################### 
################################################################### 
################################################################### 
################################################################### 
################################################################### 
################################################################### 
################################################################### 
################################################################### 
################################################################### 
################################################################### 
################################################################### 
################################################################### 
################################################################### 
################################################################### 
################################################################### 
################################################################### 
################################################################### 


class EditDisfModel:
	
	def __init__(self, model, strategy, batch_size=32):
		self.model = model
		self.strategy = strategy
		self.train_stats = None
		self.test_stats = None
		self.val_stats = None
		self.batch_size = batch_size

	def build(self, **params):
		self.model.build(**params)

	def train(self, ds_train, ds_test=None, ds_val=None, nb_epoch=20, verbose=True):
		logger.info('Training...')
		callbacks = []
		nb_train = len(ds_train.texts)
		val_tuple = None
		
		logger.info('Preparing train input...')
		train_data, train_y, train_gold = self.model.prepare_input(ds_train, self.strategy)
		self.train_stats = ModelStats(train_data, train_gold, cls=self, what='Train', verbose=verbose)
		callbacks.append(self.train_stats)

		logger.info('Calculating sample weight...')
		classes = list(range(self.model.nb_classes))
		outputs = np.array(unroll(train_gold))
		class_weight = dict(zip(classes, compute_class_weight('balanced', classes, outputs)))
		sample_weight = np.array(list(map(lambda x: list(map(class_weight.__getitem__, x)), train_gold)))
		if self.strategy.name == 'window':
			sample_weight = class_weight

		if ds_val is not None:
			logger.info('Preparing val input...')
			val_data, val_y, val_gold = self.model.prepare_input(ds_val, self.strategy)
			val_sw = np.array(list(map(lambda x: list(map(class_weight.__getitem__, x)), val_gold)))
			val_tuple = (val_data, val_y, val_sw)
			self.val_stats = ModelStats(val_data, val_gold, cls=self, what='Val', verbose=verbose)
			callbacks.append(self.val_stats)

		if ds_test is not None:
			logger.info('Preparing test input...')
			test_data, test_y, test_gold = self.model.prepare_input(ds_test, self.strategy)
			self.test_stats = ModelStats(test_data, test_gold, cls=self, what='Test', verbose=verbose)
			callbacks.append(self.test_stats)
		
		# callbacks.append(EarlyStopping(monitor='val_loss', patience=3, mode='auto'))

		logger.info('Fitting...')
		self._fit(train_data, train_y, val_data=val_tuple, sample_weight=sample_weight, 
				nb_epoch=nb_epoch, callbacks=callbacks)

		if self.strategy.name == 'dicted':
			tf, tp, tr, tpred, tgold = self.evaluate(train_data, train_gold, return_predictions=True, return_golds=True, verbose=verbose)
			self.train_stats.stats = [tf,tp,tr] # f1, prec, rec
			self.train_stats.best_pred = tpred
			self.train_stats.best_gold = tgold
			bf, bp, br, bpred, bgold = self.evaluate(test_data, test_gold, return_predictions=True, return_golds=True, verbose=verbose)
			self.test_stats.stats = [bf, bp, br] # f1, prec, rec
			self.test_stats.best_pred = bpred
			self.test_stats.best_gold = bgold


	def _fit(self, train_data, train_y, val_data=None, sample_weight=None, nb_epoch=20, callbacks=[]):
		if self.strategy.name == 'bucket':
			train_buckets 	= bucketize(train_y, max_bucket_size=self.batch_size)
			train_generator = self.model.get_batch(train_data, train_y, sample_weight, *train_buckets, shuffle=True)
			val_generator 	= None
			nb_val_samples	= None
			if val_data is not None:
				val_buckets 	= bucketize(val_data[1], max_bucket_size=self.batch_size)
				val_generator	= self.model.get_batch(*val_data, *val_buckets, shuffle=False)
				nb_val_samples	= len(val_data[1])
			self.model.classifier.fit_generator(train_generator, samples_per_epoch=len(train_buckets[0]),
												validation_data=val_generator, nb_val_samples=nb_val_samples,
												nb_epoch=nb_epoch, callbacks=callbacks)
		elif self.strategy.name == 'dicted':
			self.model.classifier.fit(train_data, train_y)			
		elif self.strategy.name == 'window':
			self.model.classifier.fit(train_data, train_y, nb_epoch=nb_epoch, batch_size=self.batch_size, 
									validation_data=val_data, callbacks=callbacks, class_weight=sample_weight)
		else:
			self.model.classifier.fit(train_data, train_y, nb_epoch=nb_epoch, batch_size=self.batch_size, 
									validation_data=val_data, callbacks=callbacks, sample_weight=sample_weight)


	def _predict_on_batch(self, generator, val_samples, return_gold=False, verbose=False):
		# !!!fix keras predict_generator function and make a pull request!!!
		# preds = self.model.classifier.predict_generator(generator, val_samples=val_samples)
		preds, golds = [], []
		if verbose:
			print('')
		for i, (X, Y, _) in enumerate(generator):
			if verbose:
				sys.stdout.write('Prediction %d/%d \r' % (i+1, val_samples))
				sys.stdout.flush()
			out = self.model.classifier.predict_on_batch(X)
			for y in out:
				preds.append(y)
			if return_gold:
				for bucket in Y.values():
					for y in bucket:
						golds.append(y)
		if return_gold:
			return preds, golds
		return preds

	def predict(self, data, verbose=False):
		if self.strategy.name == 'bucket':
			lengths, data_by_length = bucketize(data[0], max_bucket_size=self.batch_size)
			pred_generator = self.model.get_batch(data, data[0], None, lengths, data_by_length, shuffle=False, kind='predict')
			preds = self._predict_on_batch(pred_generator, len(data_by_length), verbose=verbose)
			preds = self.strategy.unprepare(preds)
			preds, _ = reorder_buckets(preds, preds, lengths, data_by_length)
			return preds, lengths, data_by_length
		elif self.strategy.name == 'dicted':
			labels = [str(l) for l in range(self.model.nb_classes)]
			flat_dict = lambda x: [[[w[l] for l in labels] for w in sent] for sent in x]
			preds = self.model.classifier.predict_marginals(data)
			preds = flat_dict(preds)
			preds = [np.array(p) for p in preds]
		else:
			preds = self.model.classifier.predict(data, batch_size=self.batch_size, verbose=verbose)
			preds = self.strategy.unprepare(preds)
		return preds

	def evaluate(self, data, gold, last_is_dot=True, return_predictions=False, return_golds=False, verbose=True):
		if self.strategy.name == 'bucket':
			lengths, data_by_length = bucketize(data[0], max_bucket_size=self.batch_size)
			pred_generator 			= self.model.get_batch(data, gold, None, lengths, data_by_length, shuffle=False, kind='predict')
			preds_vec, golds		= self._predict_on_batch(pred_generator, len(data_by_length), return_gold=True, verbose=verbose)
			preds_vec, golds 		= reorder_buckets(preds_vec, golds, lengths, data_by_length)
			preds_vec 				= self.strategy.unprepare(preds_vec)
			preds 					= [unvectorize(np.array(p)) for p in preds_vec] 
		elif self.strategy.name == 'dicted':
			golds = gold
			preds_vec = self.predict(data, verbose=verbose)
			preds = [unvectorize(p) for p in preds_vec]
		elif self.strategy.name == 'window':
			golds = gold
			preds_vec = self.predict(data, verbose=verbose)
			preds = unvectorize(np.array(preds_vec))
			preds = reshape_like(preds, map_with=golds)
			preds_vec = reshape_like(preds_vec, map_with=golds)
		else:
			golds = gold
			preds_vec = self.predict(data, verbose=verbose)
			preds = unvectorize(np.array(preds_vec))
		stats = list(Statistics.get_metrics(unroll(golds), unroll(preds)))
		plus = []
		if return_predictions:
			plus.append(preds_vec)
		if return_golds:
			plus.append(golds)
		return stats + plus

	def save_weights(self, filename):
		self.model.save_weights(filename)

	def load_weights(self, filename):
		self.model.load_weights(filename)