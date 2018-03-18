import os
import datetime

def save_to_file(word_texts, word_indexes, predictions, fname='', dname='', task='ss', write_to_file=True):
	today_date = datetime.datetime.now().strftime('%y-%m-%d')
	dname = today_date if dname == 'now' else dname
	new_texts = []

	if not os.path.exists('data/saves/%s/' % dname):
		os.mkdir('data/saves/%s/' % dname)

	for i, (text, index) in enumerate(zip(word_texts, word_indexes)):
		s = ''
		j = 0
		for word in text:
			if task == 'ss':
				if word != '.' and word != '*':
					dot = ' . ' if predictions[i][j] == 1 else ' '
					s += word + dot
					j += 1
			elif task == 'dd_fillers':
				inv_filler_scheme = {0:'.', 1:'*', 2:'+', 3:'$'}
				if word not in inv_filler_scheme.values():
					dot = ' '
					if predictions[i][j] >= 1:
						dot = ' ' + inv_filler_scheme[predictions[i][j]] + ' '
					s += word + dot
					j += 1
			elif task == 'dd_editdisfs' or task == 'dd_editdisfs_binary':
				inv_editdisf_scheme = {1:'*', 2:'+', 3:'$'}
				if word == '.':
					s += ' . '
				elif word not in inv_editdisf_scheme.values():
					dot = ' '
					if predictions[i][j] >= 1:
						dot = ' ' + inv_editdisf_scheme[predictions[i][j]] + ' '
					s += word + dot
					j += 1
			elif task == 'ssdd':
				if word != '.' and word != '*':
					dot = ' '
					if predictions[i][j] == 1:
						dot = ' . '
					elif predictions[i][j] == 2:
						dot = '* '
					elif predictions[i][j] == 3:
						dot = '* . '
					s += word + dot
					j += 1
		if write_to_file:
			f = open('data/saves/%s/%s-%d.txt' % (dname, fname, index), 'w', encoding='utf8')
			f.write(s.strip())
			f.close()
		new_texts.append(s.strip())
	return new_texts

def get_new_texts(word_texts, word_indexes, predictions, fname='', dname='', task='ss', write_to_file=True):
	return save_to_file(word_texts, word_indexes, predictions, fname=fname, dname=dname, task=task, write_to_file=False)


def convert_prediction_to_tuple(pred_text):
	symbols = ['.', '*', '+', '$']
	inner_text = []
	for token in pred_text.split():
		if token not in symbols:
			inner_text.append((token, ''))
		else:
			inner_text[-1] = (inner_text[-1][0], token)
	return inner_text

def convert_predictions_to_tuples(pred_texts):
	return [convert_prediction_to_tuple(text) for text in pred_texts]

def convert_tuple_to_text(tuple_text):
	return ' '.join(list(map(lambda x: x[0]+' '+x[1], tuple_text))).strip()

def convert_tuples_to_texts(tupled_texts):
	return [convert_tuple_to_text(text) for text in tupled_texts]
