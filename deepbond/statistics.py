import warnings
import logging
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class Statistics:

	@staticmethod
	def accuracy(x, y):
		return np.mean(x == y) * 100.0

	@staticmethod
	def mean_relative_error(x, y):
		return sum([(i-j)/j for i,j in zip(x, y)])/len(y)
	
	@staticmethod
	def get_metrics(gold, predicted):
		gold = np.array(gold)
		predicted = np.array(predicted)
		gold[gold > 0] = 1
		predicted[predicted > 0] = 1 
		p = precision_score(gold, predicted, pos_label=1, average='binary') 
		r = recall_score(gold, predicted, pos_label=1, average='binary')
		f1 = f1_score(gold, predicted, pos_label=1, average='binary')
		return f1, p, r

	@staticmethod
	def chance_baseline(gold):
		if isinstance(gold, list):
			gold = np.array(gold).flatten()
		gold[gold > 0] = 1
		predicted = np.ones(gold.shape)
		Statistics.print_metrics(gold, predicted)
		return np.mean(gold)

	@staticmethod
	def print_metrics(gold, predicted, print_cm=False):
		gold[gold > 0] = 1
		predicted[predicted > 0] = 1 
		cm = confusion_matrix(gold, predicted)
		logger.debug("CLASS NO:  Precision = %.4f  Recall = %.4f F-Measure = %.4f" % (precision_score(gold, predicted, pos_label=0, average='binary'), 
																			  recall_score(gold, predicted, pos_label=0, average='binary'),
																			  f1_score(gold, predicted, pos_label=0, average='binary')))
		
		logger.debug("CLASS YES: Precision = %.4f  Recall = %.4f F-Measure = %.4f" % (precision_score(gold, predicted, pos_label=1, average='binary'), 
																			  recall_score(gold, predicted, pos_label=1, average='binary'), 
																			  f1_score(gold, predicted, pos_label=1, average='binary'))) 
		 
		logger.debug("AVERAGE:   Precision = %.4f  Recall = %.4f F-Measure = %.4f" % (precision_score(gold, predicted, pos_label=None, average='macro'), 
																			  recall_score(gold, predicted, pos_label=None, average='macro'),
																			  f1_score(gold, predicted, pos_label=None, average='macro')))

		logger.debug("Accuracy = %.4f" % accuracy_score(gold, predicted))
		if print_cm:
			logger.debug("Confusion Matrix")
			logger.debug(cm)
		return f1_score(gold, predicted, pos_label=1, average='binary')