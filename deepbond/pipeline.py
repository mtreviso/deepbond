import os, re

class Pipeline:

	def __init__(self, sbd=None, fd=None, edd=None, verbose=True):
		self.sbd = sbd
		self.fd = fd
		self.edd = edd
		if self.sbd is not None:
			self.sbd.verbose = verbose
		if self.fd is not None:
			self.fd.verbose = verbose
		if self.edd is not None:
			self.edd.verbose = verbose
		
	@staticmethod
	def remove_fillers_listbased(preds, list_fname):
		def load_list(fname):
			l = []
			with open(fname, 'r', encoding='utf8') as f:
				for line in f:
					l.append(line.strip())		
			return l
		filler_set = set(load_list(list_fname))
		new_preds = []
		for pred in preds:
			inner_preds = []
			for word, label in pred:
				if word not in filler_set:
					inner_preds.append((word, label))
			new_preds.append(inner_preds)
		return new_preds


	@staticmethod
	def merge_ss_and_fillers(pred_ss, pred_fillers):
		new_preds = []
		for i in range(len(pred_ss)):
			inner_preds = []
			for t_ss, t_f in zip(pred_ss[i], pred_fillers[i]):
				assert(t_ss[0] == t_f[0])
				word = t_ss[0]
				label_ss = t_ss[1]
				label_f = t_f[1]
				if label_ss == '' and label_f == '':
					inner_preds.append((word, ''))
				elif label_ss != '' and label_f == '':
					inner_preds.append((word, label_ss))
				elif label_ss == '' and label_f != '':
					continue
				elif label_ss != '' and label_f != '':
					if len(inner_preds) == 0:
						continue
					elif inner_preds[-1][1] == '':
						inner_preds[-1] = (inner_preds[-1][0], label_ss)
			new_preds.append(inner_preds)
		return new_preds


	@staticmethod
	def remove_disfs(pred_disfs):
		new_preds = []
		for preds in pred_disfs:
			inner_preds = []
			for word, label in preds:
				if label == '' or label == '.':
					inner_preds.append((word, label))
			new_preds.append(inner_preds)
		return new_preds


	def fit(self, texts=[], audios=[], without_editdisfs=False):
		
		if self.sbd is not None:
			# sentence segmentation
			pred_ss = self.sbd.detect(texts, audios)
			for x in pred_ss:
				print(x)

		if self.fd is not None:
			# filler detection (md + é)
			pred_fillers = self.fd.detect(texts, [])
		
		if self.sbd is not None and self.fd is not None:
			# merge sentence boundaries and fillers predictions
			new_preds = Pipeline.merge_ss_and_fillers(pred_ss, pred_fillers)

			# detect filled pauses using a list of selected words
			new_preds = Pipeline.remove_fillers_listbased(new_preds, 'data/lists/pp.txt')
			
			# convert predictions to texts
			new_texts = [' '.join(list(map(lambda x:x[0]+' '+x[1], text))) for text in new_preds]
			new_texts = [re.sub(r'\ +', ' ', text).strip() for text in new_texts]

		if self.edd is None:
			return new_texts

		# detect edit disfluences
		pred_editdisfs = self.edd.detect(new_texts, [])

		# remove edit disfluences
		pred_editdisfs = Pipeline.remove_disfs(pred_editdisfs)

		# convert predictions to texts
		final_texts = [' '.join(list(map(lambda x:x[0]+x[1], text))) for text in pred_editdisfs]

		return final_texts

