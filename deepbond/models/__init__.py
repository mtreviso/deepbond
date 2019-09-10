from collections import defaultdict
from pathlib import Path

from deeptagger import constants
from deeptagger import opts
from .cnn import CNN
from .rnn import RNN
from .rcnn import RCNN
from .simple_lstm import SimpleLSTM


available_models = {
    'simple_lstm': SimpleLSTM,
    'rcnn': RCNN,
    'cnn': CNN,
    'rnn': RNN,
}


def build(options, fields_tuples):
    # dict_fields returns None if a field doesnt exist
    dict_fields = defaultdict(lambda: None)
    dict_fields.update(dict(fields_tuples))
    model_class = available_models[options.model]
    model = model_class(dict_fields['words'],
                        dict_fields['tags'],
                        prefixes_field=dict_fields['prefixes'],
                        suffixes_field=dict_fields['suffixes'],
                        caps_field=dict_fields['caps'])
    model.build(options)
    if options.gpu_id is not None:
        model = model.cuda(options.gpu_id)
    return model


def load_state(path, model):
    model_path = Path(path, constants.MODEL)
    model.load(str(model_path))


def load(path, fields_tuples):
    options = opts.load(path)
    model = build(options, fields_tuples)
    load_state(path, model)
    return model


def save(path, model):
    model_path = Path(path, constants.MODEL)
    model.save(model_path)
