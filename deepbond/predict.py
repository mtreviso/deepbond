import logging
from pathlib import Path

from deeptagger import constants
from deeptagger.dataset import dataset, fields
from deeptagger import features
from deeptagger import iterator
from deeptagger import models
from deeptagger.predicter import Predicter


def run(options):
    words_field = fields.WordsField()
    tags_field = fields.TagsField()
    fields_tuples = [('words', words_field), ('tags', tags_field)]
    fields_tuples += features.load(options.load)

    if options.test_path is None and options.text is None:
        raise Exception('You should inform a path to test data or a text.')

    if options.test_path is not None and options.text is not None:
        raise Exception('You cant inform both a path to test data or a text.')

    dataset_iter = None
    if options.test_path is not None:
        logging.info('Building test dataset: {}'.format(options.test_path))
        test_tuples = list(filter(lambda x: x[0] != 'tags', fields_tuples))
        test_dataset = dataset.build(options.test_path, test_tuples, options)

        logging.info('Building test iterator...')
        dataset_iter = iterator.build(test_dataset, options.gpu_id,
                                      options.dev_batch_size, is_train=False)

    if options.text is not None:
        logging.info('Preparing text...')
        test_tuples = list(filter(lambda x: x[0] != 'tags', fields_tuples))
        test_dataset = dataset.build_texts(options.text, test_tuples, options)

        logging.info('Building iterator...')
        dataset_iter = iterator.build(test_dataset, options.gpu_id,
                                      options.dev_batch_size, is_train=False)

    logging.info('Loading vocabularies...')
    fields.load_vocabs(options.load, fields_tuples)

    logging.info('Loading model...')
    model = models.load(options.load, fields_tuples)

    predicter = Predicter(dataset_iter, model)
    predictions = predicter.predict(options.prediction_type)

    if options.prediction_type == 'classes':
        prediction_tags = transform_classes_to_tags(tags_field, predictions)
        predictions_str = transform_predictions_to_text(prediction_tags)
    elif options.prediction_type == 'probas':
        predictions_str = transform_predictions_to_text(predictions)

    if options.text is None:
        save_predictions(options.output_dir, predictions_str)
    else:
        logging.info(options.text)
        logging.info(predictions_str)

    return predictions


def save_predictions(directory, predictions_str):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    output_path = Path(directory, constants.PREDICTIONS)
    output_path.write_text(predictions_str)
    logging.info('Predictions saved in {}'.format(output_path))


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
