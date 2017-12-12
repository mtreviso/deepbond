import logging
import numpy as np
from deepbond.statistics import Statistics
from deepbond.utils import unvectorize, unroll

logger = logging.getLogger(__name__)

class LinearCombiner:

	def __init__(self, lexical_stats=None, prosodic_stats=None, last_word_is_boundary=True):
		self.lexical = lexical_stats
		self.prosodic = prosodic_stats
		self.best_prediction = None
		self.best_stats = [0, 0, 0]
		self.last_word_is_boundary = last_word_is_boundary
		self.best_p = 0

		if self.prosodic is not None and self.lexical is not None:
			assert(len(self.lexical.best_pred) == len(self.prosodic.best_pred))
			assert(len(self.lexical.best_gold) == len(self.prosodic.best_gold))
			for i in range(len(self.prosodic.best_pred)):
				assert(len(self.prosodic.best_pred[i]) == len(self.lexical.best_pred[i]))
				assert(len(self.prosodic.best_gold[i]) == len(self.lexical.best_gold[i]))

		if self.prosodic is not None:
			self.gold = np.array(unroll(self.prosodic.best_gold))
			self.prosodic_prediction = np.array(self.prosodic.best_pred)
			self.best_prediction = self.prosodic_prediction
			self.best_stats = self.prosodic.stats
			self.best_p = 0.0
		
		if self.lexical is not None:
			self.gold = np.array(unroll(self.lexical.best_gold))
			self.lexical_prediction = np.array(self.lexical.best_pred)
			self.best_prediction = self.lexical_prediction
			self.best_stats = self.lexical.stats
			self.best_p = 1.0

	def combine(self, step=0.1, verbose=True):
		assert(step >= 0 and step <= 1)
		assert(self.lexical is not None and self.prosodic is not None)
		partitions = np.arange(0, 1+step, step).tolist()
		for p in partitions:
			new_pred = unroll(self._fit(p, self.lexical_prediction, self.prosodic_prediction))
			new_stats = Statistics.get_metrics(self.gold, new_pred)
			if new_stats[0] >= self.best_stats[0]:
				self.best_stats = new_stats
				self.best_prediction = new_pred
				self.best_p = p
			if verbose:
				self.show(p, *new_stats)
		if verbose:
			logger.info('Best combination: ')
			self.show(self.best_p, *self.best_stats)

	def _last_word_is_boundary(self, p):
		if self.last_word_is_boundary:
			for i in range(len(p)):
				p[i][-1] = 1
		return p

	def _fit(self, p, l_pred, p_pred):
		new_pred = p * l_pred + (1 - p) * p_pred
		new_pred = list(map(unvectorize, new_pred))
		new_pred = self._last_word_is_boundary(new_pred)
		return new_pred

	def predict(self, l_pred, p_pred):
		return self._fit(self.best_p, np.array(l_pred), np.array(p_pred))

	def evaluate(self, l_pred, p_pred, gold, verbose=True):
		new_pred = unroll(self.predict(l_pred, p_pred))
		new_gold = unroll(gold)
		stats = Statistics.get_metrics(new_gold, new_pred)
		if verbose:
			self.show(self.best_p, *stats)
		return stats

	def show(self, p, f1, prec, rec):
		logger.info('Combination for p: %.2f' % p)
		logger.info('Precision: %.4f' % prec)
		logger.info('Recall   : %.4f' % rec)
		logger.info('F-Measure: %.4f' % f1)

	def save(self, filename):
		import json
		data = {'best_p': self.best_p}
		with open(filename, 'w+') as f:
			json.dump(data, f)

	def load(self, filename):
		import json
		with open(filename, 'r') as f:
			data = json.load(f)
		self.best_p = data['best_p']