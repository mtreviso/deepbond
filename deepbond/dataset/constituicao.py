from deepbond.dataset import DataSet
from deepbond.helpers import Cleaner
from deepbond.helpers.number2word import cardinals, ordinals, letters
from os import listdir
import re

class ConstituicaoDataSet(DataSet):

	def _read_text_dir(self, dname):
		self.regex_str = []
		slash = '' if dname[-1] == '/' else '/'
		f = lambda x: int(x.split('.')[0])
		for i, fname in enumerate(sorted(listdir(dname), key=f)):
			self._read_text_file(dname + slash + fname)
		# self._eliminate_wrong_texts()

	def _eliminate_wrong_texts(self):
		l = []
		for i in range(len(self.texts)):
			if len(self.texts[i]) < 5:
				print(i)
				print(self.texts[i])
				l.append(i)
		for i in l[::-1]:
			del self.texts[i]

	def _read_text_dir_aux(self, dname):
		self._construct_regex()
		self._read_text_dir_aux(dname)
		# self._remove_correferences()
		self._save_text_dir('/'.join(dname.split('/')[:-2])+'/Constituicao-Etiquetada')
		self._merge_articles()
		self._save_text_dir('/'.join(dname.split('/')[:-2])+'/Constituicao-Artigos')
		self._transform_clues_to_dots()
		self._save_text_dir('/'.join(dname.split('/')[:-2])+'/Constituicao-Filtrada-Artigos')

	def _save_text_dir(self, dname):
		for i, t in enumerate(self.texts):
			fname = '%.4d.txt' % (i+1)
			g = open(dname+'/'+fname, 'w', encoding='utf8')
			g.write(t)
			g.close()

	def _construct_regex(self):
		# https://pt.wikipedia.org/wiki/Constituição_brasileira_de_1988
		clues =		['título', 'capítulo', 'parágrafo', 'artigo', 'inciso'] #, 'número'
		clues_nb =  [11, 		11, 		30, 		 290, 	   90] 		#, 10
		clues_sub = [' [%s] ' % s.upper() for s in clues]
		clues_pipe = ('|', '')
		self.regex_str = []
		for c, cn, cs in zip(clues, clues_nb, clues_sub):
			rs = r''
			rs += r'(?<!(\bna \b|\bno \b|\bdo \b|\bda \b|\b o \b|\b a \b|lo \b))%s' % '('
			for i, (card, ordi) in enumerate(zip(cardinals[:cn][::-1], ordinals[:cn][::-1])):
				pipe = clues_pipe[i == cn-1]
				rs += r'(\b%s %s\b)|(\b%s %s\b)%s' % (c, card, c, ordi, pipe)
			rs += ')'
			self.regex_str.append((rs, cs))
		self.regex_str.append((r'(?<!(\bna \b|\bno \b|\bdo \b|\bda \b|\b o \b|\b a \b|lo \b))(\bparágrafo único\b)', ' [PARÁGRAFO ÚNICO] '))
		self.regex_str.append((r'(?<!(\bna \b|\bno \b|\bdo \b|\bda \b|\b o \b|\b a \b|lo \b))(\balínea [a-z]\b)', ' [ALÍNEA] '))

	def _remove_lexical_clues(self, text):
		for rs, cs in self.regex_str:
			text = re.sub(rs, cs, text)
		return text

	def _remove_items(self): 
		# remover incisos e alíneas
		pass

	def _transform_clues_to_dots(self, ignore_correference_erros=True):
		for i in range(len(self.texts)):
			self.texts[i] = re.sub(r'(\[INCISO\])|(\[ALÍNEA\])|(\[PARÁGRAFO\])|(\[PARÁGRAFO ÚNICO\])', '.', self.texts[i])
			if ignore_correference_erros:
				self.texts[i] = re.sub(r'\. (\. )+', '. ', self.texts[i])

	def _merge_articles(self):
		# incisos e alíneas não viram boundaries (ignora descrições dos títulos e dos capítulos)
		key = r'(\[TÍTULO\])|(\[CAPÍTULO\])|(\[ARTIGO\])'
		new_texts = []
		first_text = ''
		last_index = 0
		last_target = 0
		for i, txt in enumerate(self.texts):
			k_indexes = [m.start() for m in re.finditer(key, txt)]
			if len(k_indexes) > 0:
				if i > 0:
					first_text = self.texts[last_target-1][last_index:]
					middle_texts = ''
					for k in range(last_target, i):
						middle_texts += self.texts[k] + ' '
					t = first_text + ' ' + middle_texts + self.texts[i][:k_indexes[0]]
					if '[TÍTULO]' not in t and '[CAPÍTULO]' not in t:
						new_texts.append(t)
				for j in range(len(k_indexes)-1):
					a, b = k_indexes[j], k_indexes[j+1]
					t = self.texts[i][a:b]
					if '[TÍTULO]' not in t and '[CAPÍTULO]' not in t:
						new_texts.append(t)
				last_index = k_indexes[-1]
				last_target = i+1
		self.texts = new_texts

	def _clean_text_file(self, text):
		text = text.replace('] .', ']')
		text = text.replace('[ARTIGO]', '')
		text = Cleaner.trim(text).strip()
		return text + ' .'

	# def _clean_text_file(self, text):
	# 	text = text.replace('ü', 'u')
	# 	text = Cleaner.remove_newlines(text)
	# 	text = Cleaner.remove_punctuation(text, less='')
	# 	text = Cleaner.trim(text)
	# 	text = self._remove_lexical_clues(text)
	# 	text = Cleaner.trim(text)
	# 	text = text.strip()
	# 	return text