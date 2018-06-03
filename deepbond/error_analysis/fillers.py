import sys
import os
import re
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score
from nltk import ngrams
from nltk.probability import FreqDist

logger = logging.getLogger(__name__)

class ErrorAnalysisFillers:

	def __init__(self, gold_dir, pred_dir, binary=True, ignore_mark=None):
		self.gold_dir = gold_dir
		self.pred_dir = pred_dir
		self.ignore_mark = ignore_mark
		self.golds, self.preds = [], []
		self.text_golds, self.text_preds = [], []

		gold_list = list(sorted(os.listdir(self.gold_dir)))
		pred_list = list(sorted(os.listdir(self.pred_dir)))
		self.binary = binary

		logger.info('Micro averaged F1: ')
		self.micro_report(gold_list, pred_list)

		logger.info('Macro averaged F1: ')
		# self.macro_report(gold_list, pred_list)

		self.unrolled_text = [w for x in self.text_golds for w in x]
		self.words = list(set(self.unrolled_text))
		self.vocab = dict(zip(self.words, range(len(self.words))))
		self.idx_vocab = dict(zip(self.vocab.values(), self.vocab.keys()))

		logger.info('Confusion Matrix: ')
		self.binary = False
		self.confusion_matrix(gold_list, pred_list)
		self.binary = binary


	def _populate(self, gold_list, pred_list):
		golds, preds = [], []
		text_golds, text_preds = [], []
		for g, p in zip(gold_list, pred_list):
			gold_text = open(os.path.join(self.gold_dir, g), 'r', encoding='utf8').read().strip()
			if 'constituicao' in self.gold_dir.lower():
				gold_text = self._clean_const(gold_text)
			if 'cachorro' in self.gold_dir.lower():
				gold_text = re.sub('(\ +)|(\-)', ' ', gold_text).strip()
			gold_labels = self._text_to_labels(gold_text)
			golds.extend(gold_labels)
			text_golds.append(gold_text.split())
			pred_text = open(os.path.join(self.pred_dir, p), 'r', encoding='utf8').read().strip()
			pred_labels = self._text_to_labels(pred_text)
			if len(pred_labels) != len(gold_labels):
				print(len(pred_labels), len(gold_labels))
				print(pred_text)
				print(gold_text)
				print('---')
			preds.extend(pred_labels)
			text_preds.append(pred_text.split())
		if len(self.golds) == 0:
			self.golds = golds
		if len(self.preds) == 0:
			self.preds = preds
		if len(self.text_golds) == 0:
			self.text_golds = text_golds
		if len(self.text_preds) == 0:
			self.text_preds = text_preds
		return golds, preds

	def _text_to_labels(self, text):
		labels = []
		word_prev = None
		list_pp = ['ahn', 'ah', 'hun', 'oh', 'hunhun', 'hum', 'heim', 
					'uhn', 'hrum', 'hinhun', 'uhm', 'hanhan', 'han', 
					'ohr', 'ha', 'eh']
		if self.binary:
			for word in text.strip().split():
				if word_prev in list_pp:
					labels[-1] = 1
				if word == '.':
					pass
				elif word == self.ignore_mark:
					pass
				elif word in '*+':
					if word_prev == 'é':
						labels[-1] = 1
				elif word == '$':
					pass
				else:
					labels.append(0)
				word_prev = word
		else:
			for word in text.strip().split():
				if word_prev in list_pp:
					labels[-1] = 1
				if word == '.':
					pass
				elif word == self.ignore_mark:
					pass
				elif word == '*':
					if word_prev == 'é':
						labels[-1] = 1
				elif word == '+':
					labels[-1] = 2
				elif word == '$':
					pass
				else:
					labels.append(0)
				word_prev = word
		return labels

	def micro_report(self, gold_list, pred_list):
		self.tp, self.tn, self.fp, self.fn = 0, 0, 0, 0
		indexes = [int(x.split('-')[-1].split('.')[0]) if '-' in x else i for i, x in enumerate(pred_list)]
		gold_list_sorted = [gold_list[i] for i in indexes]
		golds, preds = self._populate(gold_list_sorted, pred_list)
		self.raw_count(np.array(preds), np.array(golds))
		self.show_stats(*self.prf(golds, preds), *self.cer_nist(golds, preds))
		self.show_confusion_table()

	def macro_report(self, gold_list, pred_list):
		self.tp, self.tn, self.fp, self.fn = 0, 0, 0, 0
		max_fold = max([int(x.split('fold_')[1].split('-')[0]) for x in pred_list]) + 1
		gold_list_folds = [[] for i in range(max_fold)]
		pred_list_folds = [[] for i in range(max_fold)]
		stats = [0, 0, 0, 0, 0]
		for fname in pred_list:
			fold = int(fname.split('fold_')[1].split('-')[0])
			index = int(fname.split('-')[-1].split('.')[0])
			pred_list_folds[fold].append(fname)
			gold_list_folds[fold].append(gold_list[index])
		for i in range(max_fold):
			golds, preds = self._populate(gold_list_folds[i], pred_list_folds[i])
			self.raw_count(np.array(preds), np.array(golds))
			stats = [x+y for x,y in zip(list(self.prf(golds, preds))+list(self.cer_nist(golds, preds)), stats)]
		self.show_stats(*[x/max_fold for x in stats])
		self.show_confusion_table()


	def raw_count(self, predicted, gold):
		self.tp += np.sum((predicted == 1) & (gold == 1))
		self.tn += np.sum((predicted == 0) & (gold == 0))
		self.fp += np.sum((predicted == 1) & (gold == 0))
		self.fn += np.sum((predicted == 0) & (gold == 1))
		self.cer = (self.fn + self.fp)/(self.fn + self.fp + self.tn + self.tp)
		self.nist_er = (self.fn + self.fp)/(self.fn + self.tp) # SER

	def _calculate_most_frequent(self):
		remove_disfluence = lambda v: [x for x in v if x not in '.*+$']
		text = remove_disfluence(self.unrolled_text)
		self.most_tp_before = pd.Series(0, index=self.words + ['<PAD>'])
		self.most_fp_before = pd.Series(0, index=self.words + ['<PAD>'])
		self.most_fn_before = pd.Series(0, index=self.words + ['<PAD>'])
		self.most_tp_after = pd.Series(0, index=self.words + ['<PAD>'])
		self.most_fp_after = pd.Series(0, index=self.words + ['<PAD>'])
		self.most_fn_after = pd.Series(0, index=self.words + ['<PAD>'])

		for i in range(len(text)):
			word_b = text[i]
			word_a = text[i+1] if i+1 < len(text) else '<PAD>'
			if self.golds[i] == 1 and self.preds[i] == 1: # tp
				self.most_tp_before[word_b] += 1
				self.most_tp_after[word_a] += 1
			elif self.golds[i] == 0 and self.preds[i] == 1: #fp
				if word_b == 'né':
					print(' '.join(text[i-4:i+2]))
				self.most_fp_before[word_b] += 1
				self.most_fp_after[word_a] += 1
			elif self.golds[i] == 1 and self.preds[i] == 0: #fn
				self.most_fn_before[word_b] += 1
				self.most_fn_after[word_a] += 1 


	def most_frequent(self, k=10):
		self._calculate_most_frequent()
		tpb = list(self.most_tp_before.nlargest(k).iteritems())
		tpa = list(self.most_tp_after.nlargest(k).iteritems())
		fpb = list(self.most_fp_before.nlargest(k).iteritems())
		fpa = list(self.most_fp_after.nlargest(k).iteritems())
		fnb = list(self.most_fn_before.nlargest(k).iteritems())
		fna = list(self.most_fn_after.nlargest(k).iteritems())

		logger.info('Top %d most frequent hits/misses for positive class:' % k)
		logger.info('           TP before |            TP after |           FP before |            FP after |           FN before |            FN after |')
		logger.info('---------------------+'*6)
		for i in range(k):
			s = '%14s (%3d) |'*6
			logger.info(s % (*tpb[i], *tpa[i], *fpb[i], *fpa[i], *fnb[i], *fna[i]))
		logger.info('---------------------+'*6)

		tpga = pd.Series(0, index=self.words + ['<PAD>'])
		tpgb = pd.Series(0, index=self.words + ['<PAD>'])
		tppa = pd.Series(0, index=self.words + ['<PAD>'])
		tppb = pd.Series(0, index=self.words + ['<PAD>'])
		for x in self.words + ['<PAD>']:
			tpgb[x] = self.most_tp_before[x] + self.most_fn_before[x]
			tpga[x] = self.most_tp_after[x] + self.most_fn_after[x]
			tppb[x] = self.most_tp_before[x] + self.most_fp_before[x]
			tppa[x] = self.most_tp_after[x] + self.most_fp_after[x]
		tpga = list(tpga.nlargest(k).iteritems())
		tpgb = list(tpgb.nlargest(k).iteritems())
		tppa = list(tppa.nlargest(k).iteritems())
		tppb = list(tppb.nlargest(k).iteritems())

		logger.info('Top %d most frequents words after and before a disfluence: ' % k)
		logger.info('         Gold before |          Gold after |         Pred before |          Pred after |')
		logger.info('---------------------+'*4)
		for i in range(k):
			logger.info('%14s (%3d) |'*4 % (*tpgb[i], *tpga[i], *tppb[i], *tppa[i]))
		logger.info('---------------------+'*4)


	def ngram_importance(self, n=[1,2,3], k=10):
		# top k argmax P(yn=1 | w1..wn)
		# 		w1..wn
		# P(y | w1..wn) = p(w1..wn | yn=1) * p(yn) / p(w1 .. wn)
		# ---
		# p(w1 .. wn) = f(w1 .. wn) / t(w1 .. wn) 
		# p(yn) = f(yn) / t(y)
		# p(w1..wn | yn=1) = f(w1 .. wn ^ yn=1) / f(yn=1)
		remove_disfluence = lambda v: [x for x in v if x not in '.*+$']
		text = remove_disfluence(self.unrolled_text)

		y = 1
		fy_g = FreqDist(self.golds)
		fy_p = FreqDist(self.preds)

		for n_ in n:
			grams = list(ngrams(text, n_, pad_left=True))
			fwy_g = FreqDist(list(zip(grams, self.golds)))
			fwy_p = FreqDist(list(zip(grams, self.preds)))
			pwy_g = lambda w, y: fwy_g[(w, y)] / fy_g[y]
			pwy_p = lambda w, y: fwy_p[(w, y)] / fy_p[y]

			logger.info('Top %d %d-gram before disfluence: ' % (k, n_))
			vg = [(w, pwy_g(w, y)) for w in set(grams) if pwy_g(w, y) > 0]
			vg = sorted(vg, reverse=True, key=lambda x: x[1])[:k]

			vp = [(w, pwy_p(w, y)) for w in set(grams) if pwy_p(w, y) > 0]
			vp = sorted(vp, reverse=True, key=lambda x: x[1])[:k]

			logger.info('%32s | %32s |' % ('Gold', 'Pred'))
			logger.info('-'*33 +'+-' + '-'*33 +'+')
			for i in range(len(vp)):
				wg = ' '.join(vg[i][0])
				wp = ' '.join(vp[i][0])
				logger.info('%32s | %32s |' % (wg, wp))
			logger.info('-'*33 +'+-' + '-'*33 +'+')

			# index, values = [], []
			# for w, p in sorted(vp, reverse=True, key=lambda x: x[1])[:k]:
				# index.append()
				# values.append(p)
				# logger.debug(' '.join(w))
			# df = pd.Series(values, index=index)
			# df.plot(kind='bar', logy=True)
			# plt.xlabel('Word')
			# plt.ylabel('Probability')
			# plt.title('P(y | x1 ... xn)')
			# plt.show()

	

	def show_confusion_table(self):
		# https://en.wikipedia.org/wiki/F1_score#Diagnostic_Testing
		logger.debug('----------+--------------+--------------+')
		logger.debug('          |   Pred pos   |   Pred neg   |')
		logger.debug('----------+--------------+--------------+')
		logger.debug('Gold pos  |   %8d   |   %8d   |' % (self.tp, self.fn))
		logger.debug('----------+--------------+--------------+')
		logger.debug('Gold neg  |   %8d   |   %8d   |' % (self.fp, self.tn))
		logger.debug('----------+--------------+--------------+')


	def show_stats(self, p, r, f, c, n):
		logger.debug('Precision = %.4f' % p)
		logger.debug('Recall    = %.4f' % r)
		logger.debug('F-Measure = %.4f' % f)
		logger.debug('CER       = %.4f' % c)
		logger.debug('NIST      = %.2f' % n)
		logger.debug('NIST2     = %.2f' % self.nist_er)

	def prf(self, golds, preds):
		p = precision_score(golds, preds, pos_label=1, average='binary') 
		r = recall_score(golds, preds, pos_label=1, average='binary')
		f = f1_score(golds, preds, pos_label=1, average='binary')
		return p, r, f

	def cer_nist(self, golds, preds):
		preds = np.array(preds)
		golds = np.array(golds)
		slots = np.sum(golds)
		errors = np.sum(golds != preds)
		cer = errors / len(golds)
		ser = errors / slots
		return cer, ser

	def confusion_matrix(self, gold_list, pred_list):
		from pandas_ml import ConfusionMatrix
		indexes = [int(x.split('-')[-1].split('.')[0]) if '-' in x else i for i, x in enumerate(pred_list)]
		gold_list_sorted = [gold_list[i] for i in indexes]
		golds, preds = self._populate(gold_list_sorted, pred_list)
		inv_scheme = {0:'WORD', 1:'PP', 2:'MD', 3:'TEE'}
		label_list = lambda x: [inv_scheme[y] for y in x]
		confusion_matrix = ConfusionMatrix(label_list(golds), label_list(preds))
		logger.debug('\n'+str(confusion_matrix))
		
		binarize_label = lambda v, x: [int(y==x) for y in v] 
		logger.debug('PP stats: ')
		pp_golds = binarize_label(golds, 1)
		pp_preds = binarize_label(preds, 1)
		self.show_stats(*self.prf(pp_golds, pp_preds), *self.cer_nist(pp_golds, pp_preds))

		logger.debug('MD stats: ')
		md_golds = binarize_label(golds, 2)
		md_preds = binarize_label(preds, 2)
		self.show_stats(*self.prf(md_golds, md_preds), *self.cer_nist(md_golds, md_preds))

		logger.debug('---')
		logger.debug(confusion_matrix._str_stats())
		logger.debug('---')
		


if __name__ == '__main__':
	# configure logger
	logger.info = print
	logger.debug = print
	gold_dir = sys.argv[1]
	pred_dir = sys.argv[2]
	ignore_mark = sys.argv[3] if len(sys.argv) == 4 else ''
	logger.debug('Analyzing errors for gold data: {}'.format(gold_dir))
	logger.debug('Analyzing errors for pred data: {}'.format(pred_dir))
	ea = ErrorAnalysisFillers(gold_dir=gold_dir, pred_dir=pred_dir, ignore_mark=ignore_mark)
	ea.most_frequent(k=10)
	# ea.ngram_importance(n=[1, 2, 3], k=10)
