import numpy as np

from argparse import Namespace
from pathlib import Path

from deeptagger import config_utils
from deeptagger.dataset import dataset, fields
from deeptagger import features
from deeptagger import iterator
from deeptagger import models
from deeptagger import optimizer
from deeptagger import scheduler
from deeptagger import opts
from deeptagger import train
from deeptagger.predict import transform_classes_to_tags
from deeptagger.predicter import Predicter


class Tagger:

    def __init__(self, gpu_id=None):
        words_field = fields.WordsField()
        tags_field = fields.TagsField()
        self.fields_tuples = [('words', words_field), ('tags', tags_field)]
        self._loaded = False
        self.options = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.gpu_id = gpu_id

    def predict(self, texts, batch_size=32, prediction_type='classes'):
        if not self._loaded:
            raise Exception('You must load a trained model first.')

        # remove tags from the list of fields
        f_tuples = list(filter(lambda x: x[0] != 'tags', self.fields_tuples))

        # build a dataset for a list of strings
        text_dataset = dataset.build_texts(texts, f_tuples, self.options)

        # build a iterator for the new dataset
        dataset_iter = iterator.build(text_dataset, self.gpu_id, batch_size, is_train=False)  # NOQA

        # create a Predicter for this dataset
        predicter = Predicter(dataset_iter, self.model)
        predictions = predicter.predict(prediction_type)

        # return str if we received a str as input
        if isinstance(texts, str):
            return predictions[0]

        return predictions

    def predict_classes(self, texts, batch_size=32):
        return self.predict(texts, batch_size, prediction_type='classes')

    def predict_probas(self, texts, batch_size=32):
        return self.predict(texts, batch_size, prediction_type='probas')

    def transform_classes_to_tags(self, classes):
        if isinstance(classes[0], int):
            return self.transform_classes_to_tags([classes])[0]
        tags_field = self.fields_tuples[1][1]
        return transform_classes_to_tags(tags_field, classes)

    def transform_probas_to_tags(self, probas):
        if isinstance(probas[0][0], float):
            return self.transform_probas_to_tags([probas])[0]
        classes = [np.argmax(probs, axis=-1).tolist() for probs in probas]
        return self.transform_classes_to_tags(classes)

    def train(self, **kwargs):
        # if options were not loaded, we use the default ones
        if self._loaded:
            options = vars(self.options)
        else:
            options = opts.get_default_args()

        # overwrite current options with the user's arguments
        # and create a Namespace with them
        options.update(kwargs)
        options = Namespace(**options)

        # configure stuff
        options.gpu_id = self.gpu_id
        options.output_dir = config_utils.configure_output(options.output_dir)
        config_utils.configure_logger(options.debug, options.output_dir)
        config_utils.configure_seed(options.seed)
        config_utils.configure_device(options.gpu_id)

        self.options = options

        # train!
        fields_tuples, model, optimizer, scheduler = train.run(self.options)
        self.fields_tuples = fields_tuples
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        # the tagger can be considered loaded
        self._loaded = True

    def load(self, dir_path):
        # load options from the json file
        self.options = opts.load(dir_path)

        # append loaded fields_tuples
        self.fields_tuples += features.load(dir_path)

        # load vocabularies for each field
        fields.load_vocabs(dir_path, self.fields_tuples)

        # set the current gpu
        self.options.gpu_id = self.gpu_id

        # load model, optimizer and scheduler
        self.model = models.load(dir_path, self.fields_tuples)
        self.optimizer = optimizer.load(dir_path, self.model.parameters())
        self.scheduler = scheduler.load(dir_path, self.optimizer)

        # now we have a loaded tagger
        self._loaded = True

    def save(self, dir_path):
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        opts.save(dir_path, self.options)
        fields.save_vocabs(dir_path, self.fields_tuples)
        models.save(dir_path, self.model)
        optimizer.save(dir_path, self.optimizer)
        scheduler.save(dir_path, self.scheduler)
