import os, re
from deepbond import constants
from deepbond import iterator
from deepbond import models
from deepbond.dataset import dataset, fields
from deepbond.predicter import Predicter
from deepbond import opts
import argparse
import logging

logger = logging.getLogger(__name__)

class SentenceBoundaryDetector(object):

	def __init__(self,text=None,test_path = None, load='', gpu_id=0, pred_type = "classes"  **kwargs):
		# 
		parser = argparse.ArgumentParser(description='DeepBond')
		opts.general_opts(parser)
		opts.preprocess_opts(parser)
		opts.model_opts(parser)
		opts.train_opts(parser)
		opts.predict_opts(parser)

		options = parser.parse_args()

		# set predict opts
		options.text = text
		options.gpu_id = gpu_id
		options.prediction_type = pred_type
		options.load = load 

		words_field = fields.WordsField()
		tags_field = fields.TagsField()
		fields_tuples = [('words', words_field), ('tags', tags_field)]

		dataset_iter = None
		save_dir_path = None

		if options.test_path is None and options.text is None:
        	raise Exception('You should inform a path to test data or a text.')

		if options.test_path is not None and options.text is not None:
			raise Exception('You cant inform both a path to test data and a text.')


		if options.test_path is not None and options.text is None:
			logger.info('Building test dataset: {}'.format(options.test_path))
			test_tuples = list(filter(lambda x: x[0] != 'tags', fields_tuples))
			test_dataset = dataset.build(options.test_path, test_tuples, options)

			logger.info('Building test iterator...')
			dataset_iter = iterator.build(test_dataset, options.gpu_id,
										options.dev_batch_size, is_train=False)
			save_dir_path = options.test_path

		if options.text is not None and options.test_path is None:
			logger.info('Preparing text...')
			test_tuples = list(filter(lambda x: x[0] != 'tags', fields_tuples))
			test_dataset = dataset.build_texts(options.text, test_tuples, options)

			logger.info('Building iterator...')
			dataset_iter = iterator.build(test_dataset, options.gpu_id,options.dev_batch_size, is_train=False)
			save_dir_path = None



		logger.info('Loading vocabularies...')
		fields.load_vocabs(options.load, fields_tuples)

		logger.info('Loading model...')
		model = models.load(options.load, fields_tuples, options.gpu_id)

		logger.info('Predicting...')
		predicter = Predicter(dataset_iter, model)
		predictions = predicter.predict(options.prediction_type)

		logger.info('Preparing to save...')
		if options.prediction_type == 'classes':
			prediction_tags = transform_classes_to_tags(tags_field, predictions)
			predictions_str = transform_predictions_to_text(prediction_tags)
		else:
			predictions_str = transform_predictions_to_text(predictions)
        
		return predictions_str



def transform_classes_to_tags(tags_field, predictions):
    tagged_predicitons = []
    for preds in predictions:
        tags_preds = [tags_field.vocab.itos[c] for c in preds]
        tagged_predicitons.append(tags_preds)
    return tagged_predicitons


def transform_predictions_to_text(predictions):
    text = []
    is_prob = isinstance(predictions[0][0], list)
    for pred in predictions:
        sentence = []
        for p in pred:
            if is_prob:
                sentence.append(', '.join(['%.8f' % c for c in p]))
            else:
                sentence.append(p)
        if is_prob:
            text.append(' | '.join(sentence))
        else:
            text.append(' '.join(sentence))
    return '\n'.join(text)


	

def join_words_and_labels(words, labels):
    new_text = []
    assert len(words) == len(labels)
    for word, label in zip(words, labels):
        if label == '_':
            new_text.append(word)
        else:
            new_text.append(word)
            new_text.append(label)
    return new_text