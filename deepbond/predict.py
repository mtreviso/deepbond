import logging
from pathlib import Path

from deepbond import constants
from deepbond import iterator
from deepbond import models
from deepbond.dataset import dataset, fields
from deepbond.predicter import Predicter


def run(options):
    words_field = fields.WordsField()
    tags_field = fields.TagsField()
    fields_tuples = [('words', words_field), ('tags', tags_field)]

    # fields_tuples += features.load(options.load)

    if options.test_path is None and options.text is None:
        raise Exception('You should inform a path to test data or a text.')

    if options.test_path is not None and options.text is not None:
        raise Exception('You cant inform both a path to test data and a text.')

    dataset_iter = None
    save_dir_path = None

    if options.test_path is not None and options.text is None:
        logging.info('Building test dataset: {}'.format(options.test_path))
        test_tuples = list(filter(lambda x: x[0] != 'tags', fields_tuples))
        test_dataset = dataset.build(options.test_path, test_tuples, options)

        logging.info('Building test iterator...')
        dataset_iter = iterator.build(test_dataset, options.gpu_id,
                                      options.dev_batch_size, is_train=False)
        save_dir_path = options.test_path

    if options.text is not None and options.test_path is None:
        logging.info('Preparing text...')
        test_tuples = list(filter(lambda x: x[0] != 'tags', fields_tuples))
        test_dataset = dataset.build_texts(options.text, test_tuples, options)

        logging.info('Building iterator...')
        dataset_iter = iterator.build(test_dataset, options.gpu_id,
                                      options.dev_batch_size, is_train=False)

        save_dir_path = None

    logging.info('Loading vocabularies...')
    fields.load_vocabs(options.load, fields_tuples)

    logging.info('Loading model...')
    model = models.load(options.load, fields_tuples)

    logging.info('Predicting...')
    predicter = Predicter(dataset_iter, model)
    predictions = predicter.predict(options.prediction_type)

    logging.info('Preparing to save...')
    if options.prediction_type == 'classes':
        prediction_tags = transform_classes_to_tags(tags_field, predictions)
        predictions_str = transform_predictions_to_text(prediction_tags)
    else:
        predictions_str = transform_predictions_to_text(predictions)

    if options.test_path is not None:
        save_predictions(
            options.output_dir,
            predictions_str,
            save_dir_path=save_dir_path,
        )
    else:
        logging.info(options.text)
        logging.info(predictions_str)

    return predictions


def save_predictions(directory, predictions_str, save_dir_path=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if save_dir_path is not None:
        output_path = Path(directory, constants.PREDICTIONS.split('.')[0])
        output_path.mkdir(exist_ok=True)
        file_names = [f.name for f in sorted(Path(save_dir_path).iterdir())]
        save_predictions_in_a_dir(output_path, file_names, predictions_str)
    else:
        output_path = Path(directory, constants.PREDICTIONS)
        save_predictions_in_a_file(output_path, predictions_str)

    logging.info('Predictions saved in {}'.format(output_path))


def save_predictions_in_a_file(output_file_path, predictions_str):
    output_file_path.write_text(predictions_str)


def save_predictions_in_a_dir(ourpur_dir_path, file_names, predictions_str):
    assert ourpur_dir_path.is_dir()
    predictions_for_each_file = predictions_str.split('\n')
    for f_name, pred_str in zip(file_names, predictions_for_each_file):
        output_path = Path(ourpur_dir_path, f_name)
        output_path.write_text(pred_str)


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
