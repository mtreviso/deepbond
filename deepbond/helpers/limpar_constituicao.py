import re
import pandas as pd
from number2word import cardinals, ordinals, letters


def save(df, name='constituicao'):
	df.to_csv(name+'.csv', index=False)

def construct_regex():
	# https://pt.wikipedia.org/wiki/Constituição_brasileira_de_1988
	clues =		['título', 'capítulo', 'parágrafo', 'artigo', 'inciso'] #, 'número'
	clues_nb =  [11, 		11, 		30, 		 290, 	   90] 		#, 10
	clues_sub = [' [%s] ' % s.upper() for s in clues]
	clues_pipe = ('|', '')
	regex_str = []
	for c, cn, cs in zip(clues, clues_nb, clues_sub):
		rs = r''
		rs += r'(?<!(\bna \b|\bno \b|\bdo \b|\bda \b|\b o \b|\b a \b|lo \b))%s' % '('
		for i, (card, ordi) in enumerate(zip(cardinals[:cn][::-1], ordinals[:cn][::-1])):
			pipe = clues_pipe[i == cn-1]
			rs += r'(\b%s %s\b)|(\b%s %s\b)%s' % (c, card, c, ordi, pipe)
		rs += ')'
		regex_str.append((rs, cs))
	regex_str.append((r'(?<!(\bna \b|\bno \b|\bdo \b|\bda \b|\b o \b|\b a \b|lo \b))(\bparágrafo único\b)', ' [PARÁGRAFO ÚNICO] '))
	regex_str.append((r'(?<!(\bna \b|\bno \b|\bdo \b|\bda \b|\b o \b|\b a \b|lo \b))(\balínea [a-z]\b)', ' [ALÍNEA] '))
	return regex_str

def merge_articles(df, sep, dels):
	last_sep = 0
	j = 1
	for i, s in enumerate(sep):
		if df.loc[s, 'word'] in ['título', 'capítulo']:
			dels.extend(list(range(s, sep[i+1])))
		else:
			st = len(sep)-1 if i+1>=len(sep) else i+1
			try:
				df.loc[s:sep[st], 'Filename'] = j
				j += 1
			except:
				print(len(sep), st)
		last_sep = s
	return df, dels

def remove_lines(df, pos, pos_n, text, indexes):
	# what = ['título', 'capítulo', 'parágrafo', 'artigo', 'inciso', 'parágrafo único', 'alínea']
	separators = ['título', 'capítulo', 'artigo']
	sep_indexes = []
	del_indexes = []
	for i, ind in enumerate(indexes):
		for ini, end in ind:
			t1 = text[:end].count(' ') + 1
			t0 = t1 - text[ini:end].count(' ') - 1
			ini = pos[t0]
			end = pos_n if t1 >= len(pos) else pos[t1]
			del_indexes.extend(list(range(ini, end)))
			if ini < len(df) and df.iloc[ini]['word'] in separators:
				sep_indexes.append(ini)
	return del_indexes, sep_indexes

def trim_sp(df):
	del_indexes = []
	for i in range(len(df)-1):
		if df.iloc[i]['word'] == 'sp' and df.iloc[i+1]['word'] == 'sp':
			del_indexes.append(df.index[i])
	return del_indexes

def remove_extra_corr(indexes, text, pos, pos_n, i):
	d = []
	PARAG, ART, INC = 2, 3, 4
	t = 50
	for X in [PARAG, INC]:
		for pini, pend in indexes[X]:
			if pini > 0:
				if 'artigo' in text[pini-t:pini]:
					print('DELETEEEEEEEED %d' % X)
					t1 = text[:pend].count(' ') + 1
					t0 = t1 - text[pini:pend].count(' ') - 1
					ini = pos[t0]
					end = pos_n if t1 >= len(pos) else pos[t1]
					print(text.count(' ')+1, len(pos))
					print(pini, pend)
					print(t0, t1)
					print(ini, end)
					d.extend(list(range(ini, end)))
				# d.extend(list(range(pini, pend)))
	return d

def remove_lexical_clues(df, regex_str):
	pros_texts = list(pros.groupby('Filename'))
	sep_indexes = []
	del_indexes = []
	j = 0
	for i, t in pros_texts:
		print('%d/%d' % (i, len(pros_texts)))
		f = []
		c = ~t['word'].isnull() & ~t['word'].isin(['sp'])
		text = ' '.join(list(t[c]['word']))
		text = text.replace('ü', 'u')
		for rs, cs in regex_str:
			f.append([(m.start(), m.end()) for m in re.finditer(rs, text)])
		pos_n = 0
		if j+1 < len(pros_texts):
			new_t = pros_texts[j+1][1]
			new_c = ~new_t['word'].isnull() & ~new_t['word'].isin(['sp'])
			pos_n = new_t[new_c].index[0]
			del_indexes.extend(remove_extra_corr(f, text, t[c].index, pos_n, i))
		dels, sep = remove_lines(df, t[c].index, pos_n, text, f)
		del_indexes.extend(dels)
		sep_indexes.extend(sep)
		j += 1
	df, del_indexes = merge_articles(df, sep_indexes, del_indexes)
	df = df.drop(del_indexes)
	df = df.drop(trim_sp(df))
	return df

def transform_clues_to_empty(texts, ignore_correference_erros=True):
	for i in range(len(texts)):
		texts[i] = re.sub(r'(\[INCISO\])|(\[ALÍNEA\])|(\[PARÁGRAFO\])|(\[PARÁGRAFO ÚNICO\])', '.', texts[i])
		if ignore_correference_erros:
			texts[i] = re.sub(r'\. (\. )+', '. ', texts[i])
	return texts

dirname = '../../data/prosodic/'
filename = dirname+'constituicao.csv'
pros = pd.read_csv(filename, encoding='utf8')

regex_str = construct_regex()
new_pros = remove_lexical_clues(pros, regex_str)
save(new_pros, dirname+'constituicao_filtrada_artigos_trimed')
