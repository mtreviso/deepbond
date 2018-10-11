import argparse
import string
import re


def clean(txt):
	txt = txt.lower().strip()
	txt = re.sub(r'\n', ' ', txt)
	less = '-'
	p = ''.join([c if c not in less else '' for c in string.punctuation])
	txt = txt.translate(txt.maketrans('', '', p))
	txt = re.sub(r' +', ' ', txt).strip()
	return txt


def read_input(input_file):
	texts = []
	with open(input_file, 'r', encoding='utf8') as f:
		for line in f:
			if line.strip():
				texts.append(clean(line))
	return texts


def save_output(output_file, preds):
	f = open(output, 'w')
	for p in preds:
		tagged_text = ' '.join(list(map(lambda x: x[0]+x[1], p)))
		f.write(tagged_text + '\n')
	f.close()


parser = argparse.ArgumentParser(description='Apply DeepBonDD for a specifc task.')
parser.add_argument('-i', '--input', type=str, help='Path to an input text file where each line is a sample.', required=True)
parser.add_argument('-o', '--output', type=str, help='Name of the generated output file.', required=True)
parser.add_argument('-t', '--task', type=str, choices=['boundaries', 'fillers', 'editdisfs'], 
					help='Select what to detect: boundaries, fillers or editdisfs', required=True)
options = parser.parse_args()


texts = read_input(options.input)
preds = None

if options.task == 'boundaries':
	from deepbond.task import SentenceBoundaryDetector
	det = SentenceBoundaryDetector(l_model='rcnn', p_model='none', verbose=True)
	det.set_model_id('SENTENCE_BOUNDARY_MODEL_FOR_CINDERELA')
	preds = det.detect(texts=texts)
elif options.task == 'fillers':
	from deepbond.task import FillerDetector
	det = FillerDetector(l_model='rcnn', p_model='none', verbose=True)
	det.set_model_id('FILLERS_MODEL_FOR_CINDERELA')
	preds = det.detect(texts=texts)
else:
	from deepbond.task import EditDisfDetector
	det = EditDisfDetector(l_model='rcnn', p_model='none', verbose=True)
	det.set_model_id('EDITDISFS_MODEL_FOR_CINDERELA')
	preds = det.detect(texts=texts)

save_output(options.output, preds)
