import logging
from keras.callbacks import Callback, EarlyStopping
from deepbond.statistics import Statistics
from deepbond.utils import unroll

logger = logging.getLogger(__name__)



class EarlyStopping(EarlyStopping):
	pass

class ModelStats(Callback):
	def __init__(self, data, gold, what='Train', cls=None, verbose=True):
		self.data = data
		self.gold = gold
		self.cls = cls
		self.stats = [0, 0, 0] # f1, prec, rec
		self.best_pred = None
		self.best_gold = None
		self.what = what
		self.verbose = verbose
		if self.verbose:
			self._input_info(data, gold, what=what)

	def on_epoch_end(self, epoch, logs={}):
		f1, prec, rec, pred, gold = self.cls.evaluate(self.data, self.gold, return_predictions=True, return_golds=True)
		if f1 > self.stats[0]:
			# just for debugging -> stats best_* are updated at the end of each epoch
			name = self.cls.model.name
			if self.what == 'Test':
				logger.info('Saving temp weights for %s model...' % name)
				self.cls.save_weights('data/models/%s_%s_weights.h5' % (self.what.lower(), name))
			elif self.what == 'Train':
				logger.info('Saving train weights for %s model... (epoch %d)' % (name, epoch))
				self.cls.save_weights('data/models/%s_%s_weights.h5' % (self.what.lower(), name))
				self.cls.save_weights('data/models/%s_%s_e%d_weights.h5' % (self.what.lower(), name, epoch))
			elif self.what == 'Val':
				logger.info('Saving val weights for %s model...' % name)
				self.cls.save_weights('data/models/%s_%s_weights.h5' % (self.what.lower(), name))
			self.stats = [f1, prec, rec]
		self.best_pred = pred 
		self.best_gold = gold
		if self.verbose:
			logger.info('Statistics for %s:' % self.what)
			self._show_statistics(f1, prec, rec)

	def _input_info(self, data, gold, what='Train'):
		logger.info('X_%s: %s' % (what, ', '.join(str(len(x)) for x in data)))
		logger.info('Y_%s: %s' % (what, str(len(gold))))
		Statistics.chance_baseline(unroll(gold))

	def _show_statistics(self, f1, prec, rec):
		logger.info('Precision: %.4f / %.4f' % (prec, self.stats[1]))
		logger.info('Recall   : %.4f / %.4f' % (rec, self.stats[2]))
		logger.info('F-Measure: %.4f / %.4f' % (f1, self.stats[0]))

