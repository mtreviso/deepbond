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
	f = open(output_file, 'w')
	for p in preds:
		f.write(p + '\n')
	f.close()


parser = argparse.ArgumentParser(description='Apply the entire DeepBonDD pipeline.')
parser.add_argument('-i', '--input', type=str, help='Path to an input text file where each line is a sample.', required=True)
parser.add_argument('-o', '--output', type=str, help='Name of the generated output file.', required=True)
options = parser.parse_args()


texts = read_input(options.input)

from deepbond.task import SentenceBoundaryDetector, FillerDetector, EditDisfDetector
from deepbond import Pipeline
	
sbd = SentenceBoundaryDetector(l_model='rcnn', p_model='none', verbose=True)
sbd.set_model_id('SENTENCE_BOUNDARY_MODEL_FOR_CINDERELA')

fd = FillerDetector(l_model='rcnn', p_model='none', verbose=True)
fd.set_model_id('FILLERS_MODEL_FOR_CINDERELA')
fd.restrict_wordset()

edd = EditDisfDetector(l_model='rcnn_crf', p_model='none', verbose=True)
edd.set_model_id('EDITDISFS_MODEL_FOR_CINDERELA')

p = Pipeline(sbd, fd, edd, verbose=False)
new_texts = p.fit(texts=texts)

save_output(options.output, new_texts)
