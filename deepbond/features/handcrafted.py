import numpy as np
from deepbond.features import Affixes

class HandCrafted:

	def __init__(self, wang_single_limit=4, wang_pair_limit=4, prefix_size=(2,4), prefix_limit=2, eos_limit=2, use_pos=False):
		self.wang_single_limit = 4
		self.wang_pair_limit = 4
		self.prefix_size = prefix_size
		self.prefix_limit = prefix_limit
		self.eos_limit = eos_limit
		
		# can be prefix or suffix
		self.affixes = Affixes(start=prefix_size[0], end=prefix_size[1])
		
		self.dimensions = wang_single_limit*2 + wang_pair_limit*2 + self.affixes.size*prefix_limit + eos_limit*2
		if use_pos:
			self.dimensions += wang_single_limit*2 + wang_pair_limit*2

		# load a trained LM model
		self.LM = None

	def get_similarity(self, word_texts):
		# edit distance?
		pass

	def get_LM_scores(self, word_texts):
		# train a LM and use it here
		pass

	def get_prefix(self, word_texts):
		'''
		Prefix(wi,k) = wi == p[wi+k] | 1 <= k <= 2
		'''
		prefixes = []
		for text in word_texts:
			text_d = []
			for i in range(len(text)):
				if text[i] == '.':
					continue
				d = []
				for k in range(1, self.prefix_limit+1):
					j = i+k
					if j >= len(text):
						for _ in range(self.affixes.size):
							d.append(0.0)
					else:
						word = text[i]
						for p in self.affixes.extract(text[j], which='prefix'):
							d.append(float(word == p))
				text_d.append(d)
			prefixes.append(text_d)
		return prefixes

	def get_eos(self, word_texts):
		'''
		EOS(wi,k) = wi+k == . | -2 <= k <= 2
		'''
		eoss = []
		for text in word_texts:
			text_d = []
			for i in range(len(text)):
				if text[i] == '.':
					continue
				d = []
				for k in range(-self.eos_limit, self.eos_limit+1):
					j = i+k
					if k == 0:
						continue
					if j < 0 or j >= len(text):
						d.append(0.0)
					else:
						d.append(float(text[j] == '.'))
				text_d.append(d)
			eoss.append(text_d)
		return eoss

	def _duplicate_window(self, word_texts, limit=5, n=1):
		dups = []
		for text in word_texts:
			text_d = []
			for i in range(len(text)):
				if text[i] == '.':
					continue
				d = []
				for k in range(-limit, limit+1):
					j = i+k
					if k == 0:
						continue
					if j < 0 or j >= len(text):
						d.append(0.0)
					else:
						c1 = text[i:i+n] == text[j:j+n]
						c2 = '.' not in text[i:i+n] and '.' not in text[j:j+n]
						d.append(float(c1 and c2))
				text_d.append(d)
			dups.append(text_d)
		return dups

	def get_wang(self, word_texts, pos_texts):
		# according to: 
		# Wang et al. (2017) Transition-Based Disfluency Detection using LSTMs
		# Duplicate(i,wi+k),−15 ≤ k ≤ +15 and k ?= 0: if wi equals wi+k, the value is 1, others 0 
		# Duplicate(pi, pi+k),−15 ≤ k ≤ +15 and k ?= 0: if pi equals pi+k, the value is 1, others 0 
		# Duplicate(wi wi+1,w i+k wi+k+1),−4 ≤ k ≤ +4 and k ?= 0: if wiwi+1 equals wi+kwi+k+1, the value is 1, others 0
		# Duplicate(pipi+1, pi+kpi+k+1), −4 ≤ k ≤ +4 and k ?= 0: if pipi+1 equals pi+kpi+k+1, the value is 1, others 0
		dups_single = self._duplicate_window(word_texts, limit=self.wang_single_limit, n=1)
		dups_pair = self._duplicate_window(word_texts, limit=self.wang_pair_limit, n=2)
		if len(pos_texts) == 0:
			return dups_single, dups_pair
		dups_single_pos = self._duplicate_window(pos_texts, limit=self.wang_single_limit, n=1)
		dups_pair_pos = self._duplicate_window(pos_texts, limit=self.wang_pair_limit, n=2)
		return dups_single, dups_pair, dups_single_pos, dups_pair_pos

	def merge(self, hc_feats):
		feats_all = []
		for i in range(len(hc_feats[0])):
			d = []
			for j in range(len(hc_feats[0][i])):
				m = []
				for f in range(len(hc_feats)):
					m.extend(hc_feats[f][i][j])
				d.append(m)
			feats_all.append(d)
		return feats_all

	def get(self, word_texts, pos_texts):
		DISF_SYMBOLS = '*+$'
		cleaned_word_texts = list(map(lambda t: list(filter(lambda x: x not in DISF_SYMBOLS, t)), word_texts))
		f1 = self.get_wang(cleaned_word_texts, pos_texts)
		f2 = self.get_prefix(cleaned_word_texts)
		f3 = self.get_eos(cleaned_word_texts)
		return self.merge([*f1, f2, f3])

