import numpy as np

class Prosodic:

	def __init__(self, type='principal', classify=True, nb_features=3, first=1, last=3, max_size=10, pad_value=0, use_pause=True):
		self.type = type
		self.classify = classify
		self.nb_features = nb_features # 3 = duration, energy and pitch
		self.nb_first = first 			# first phone
		self.nb_last = last 			# last three phones (+ short pause)
		self.max_size = max_size 		# max phones for a word
		self.pad_value = pad_value 	# pad with -1 until max_size
		self.nb_phones =  self.nb_first + self.nb_last + int(use_pause)
		if self.type == 'padding':
			self.nb_phones = self.max_size

	def _apply_prosodic_strategy(self, phones_features):
		fts = []
		if self.type == 'principal':
			for i in range(self.nb_first+1):
				if i < len(phones_features):
					fts.append(phones_features[i])
				else:
					fts.append(phones_features[-1])
			for i in range(self.nb_last, 0, -1):
				if i < len(phones_features):
					fts.append(phones_features[-i])
				else:
					fts.append(phones_features[-1])
		elif self.type == 'padding':
			s = self.max_size - len(phones_features)
			for _ in range(s):
				phones_features.append([self.pad_value] * self.nb_features)
			fts = phones_features
		else:
			raise Exception('Strategy %s not implemented.' % self.type)
		unrolled_fts = [y for x in fts for y in x]
		return unrolled_fts

	def _apply_prosodic_transformation(self, phones_features, avgs):
		# pf shape 		[pause first last-3 last-2 last-1]
		# avgs shape 	[first last-3 last-2 last-1]
		start_phones_index = self.nb_features
		for i in range(len(avgs)):
			phones_features[i+start_phones_index] = int(phones_features[i+start_phones_index] > avgs[i])
		return phones_features

	def calc_avg_last_vowel(self, j, content):
		t = [0 for _ in range(self.nb_first + self.nb_last)]
		sums = [0 for _ in range(self.nb_features*(self.nb_first + self.nb_last))]
		last_k = 0
		for k in range(j, len(content)):
			word = content.iloc[k]['word']
			if word != None and word != 'sp':
				s = int(content.iloc[k-1]['word'] == 'sp' and k > 0)
				for i in range(self.nb_first):
					sums[i*self.nb_features + 0] += float(content.iloc[k]['Duration'])
					sums[i*self.nb_features + 1] += float(content.iloc[k]['Intensity'])
					sums[i*self.nb_features + 2] += float(content.iloc[k]['fo2'])
					t[i] += 1
				for i in range(self.nb_last):
					l = self.nb_first + i
					m = self.nb_last - i
					if k-s-m >= j and k-s-m > last_k:
						sums[l*self.nb_features + 0] = float(content.iloc[k-s-m]['Duration'])
						sums[l*self.nb_features + 1] = float(content.iloc[k-s-m]['Intensity'])
						sums[l*self.nb_features + 2] = float(content.iloc[k-s-m]['fo2'])
						t[l] += 1
				last_k = k
		avgs = [x / t[i // self.nb_features] for i, x in enumerate(sums)]
		return avgs

	def get(self, pros_texts, mask_lines=None, mask_value=0.0, average=False, normalize=True):
		prosodic = []
		# Consoantes naovozeadas (nao pode usar nenhum feature):
		nao_vozeadas = ['ff', 'kk', 'pp', 'rd', 'rr', 'sh', 'ss', 'tt', 'ts']
		# Consoantes vozeadas(pode usar F0):
		vozeadas = ['bb', 'dd', 'dz', 'gg', 'mm', 'nh', 'nn', 'vv', 'zh', 'zz']
		# Vogais (pode usar duracao intensidade e F0):
		vogais = ['aa', 'an', 'ao', 'ee', 'eh', 'en', 'ii', 'in', 'lh', 'll', 'on', 'oo', 'un', 'uu']
		for i, data in enumerate(pros_texts):
			if not data:
				if mask_lines is None:
					prosodic.append(None)
				else:
					s = self.nb_features * self.nb_phones
					masked_matrix = [[mask_value for _ in range(s)] for _ in range(mask_lines[i])]
					prosodic.append(masked_matrix)
				continue
			fname, content = data
			j = 0
			while content.iloc[j]['word'] == 'sp' or content.iloc[j]['word'] == None:
				j += 1
			p, feats = [], []
			has_sp = False
			for k in range(j, len(content)):
				phone = content.iloc[k]['Phone']
				word = content.iloc[k]['word']
				duration = float(content.iloc[k]['Duration'])
				intensity = float(content.iloc[k]['Intensity'])
				f02 = float(content.iloc[k]['fo2'])
				fts = [duration, intensity, f02]
				if self.classify:
					if phone in nao_vozeadas:
						fts = [0.0, 0.0, 0.0]
					elif phone in vozeadas:
						fts = [0.0, 0.0, f02]
					else:
						fts = [duration, intensity, f02]
				if word == 'sp':
					has_sp = True
					fts = [fts[0], 0.0, 0.0]
				feats.append(fts)
				if word != 'sp' and word != None and k > j:
					first_phone = feats.pop(-1)
					if not has_sp:
						feats.append([0.0, 0.0, 0.0])
					p.append(self._apply_prosodic_strategy(feats[:-1]))
					feats = [feats[-1], first_phone]
					has_sp = False
			if len(feats) > 0:
				if not has_sp:
					feats.append([0.0, 0.0, 0.0])
				p.append(self._apply_prosodic_strategy(feats))
			# prosodic shape (nb_texts, nb_words, nb_phones*3)
			avgs = None
			if normalize:
				sums = np.zeros(len(p[0]))
				stds = np.zeros(len(p[0]))
				for word in p:
					sums += np.array(word)
				avgs = sums / len(p)
				for word in p:
					stds += np.power(np.array(word) - avgs, 2)
				stds = np.sqrt(stds / len(p))
				for k in range(len(p)):
					p[k] = (p[k] - avgs)/stds
			if average:
				if avgs is None:
					sums = np.zeros(len(p[0]))
					for word in p:
						sums += np.array(word)
					avgs = sums / len(p)
				for k in range(len(p)):
					for i in range(len(p[i])):
						p[k][i] = int(p[k][i] > avgs[i])
			prosodic.append(p)
		return prosodic

	def test_prosodic(self, pros_texts, map_with):
		pros = self.get(pros_texts)
		t = 0
		print(len(pros), len(map_with))
		for i, (a, b) in enumerate(zip(pros, map_with)):
			if len(a) != len(b) - b.count('.') - b.count('*'):
				# for x in pros[i]:
				# 	print(x)
				t += 1
				print('Dimension mismatch on file %d: %d vs %d | %d' % (i+1, len(a), len(b)-b.count('.')-b.count('*'), b.count('.')+b.count('*')))
				print(b)
				# raise Exception('Dimension mismatch on file %d: %d vs %d - %d' % (i+1, len(a), len(b), b.count('.')))
		print('Total errors: %d' % t)

	def _divide_prosodic(self, prosodic):
		"""
		divide into 4 arrays with shape:
		duration_pause with shape (nb_words, 1)
		duration_phones with shape (nb_words, nb_phones)
		pitch with shape (nb_words, nb_phones)
		intensity with shape (nb_words, nb_phones)
		:return: [duration_pause, pitch, intensity] 
		"""
		durations_pauses = []
		durations = []
		pitches = []
		intensities = []
		# prosodic shape (nb_texts, nb_words, nb_phones*3)
		for text in prosodic:
			dp, d, p, i = [], [], [], []
			for word in text:
				x, y, z = [], [], []
				for k in range(1, self.nb_phones):
					phone = word[(k-1)*self.nb_features:k*self.nb_features]
					duration, intensity, pitch = phone
					x.append(duration)
					y.append(intensity)
					z.append(pitch)
				dp.append([word[-self.nb_features]])
				d.append(x)
				p.append(y)
				i.append(z)
			durations_pauses.append(dp)
			durations.append(d)
			pitches.append(p)
			intensities.append(i)
		return durations_pauses, durations, pitches, intensities
