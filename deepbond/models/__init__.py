from collections import defaultdict
from pathlib import Path

from deepbond import constants
from deepbond import opts
from .cnn import CNN
from .rnn import RNN
from .rcnn import RCNN


available_models = {
    'rcnn': RCNN,
    'cnn': CNN,
    'rnn': RNN,
}


def build(options, fields_tuples, loss_weights):
    # dict_fields returns None if a field doesnt exist
    dict_fields = defaultdict(lambda: None)
    dict_fields.update(dict(fields_tuples))
    model_class = available_models[options.model]
    model = model_class(
        dict_fields['words'],
        dict_fields['tags'],
        options
    )
    model.build_loss(loss_weights)
    if options.gpu_id is not None:
        model = model.cuda(options.gpu_id)
    return model


def load_state(path, model):
    model_path = Path(path, constants.MODEL)
    model.load(model_path)


def load(path, fields_tuples, current_gpu_id):
    options = opts.load(path)

    # set gpu device to the current device
    options.gpu_id = current_gpu_id

    # hack: set dummy loss_weights (the correct values are going to be loaded)
    tags_field = dict(fields_tuples)['tags']
    loss_weights = None
    if options.loss_weights == 'balanced':
        loss_weights = [0] * (len(tags_field.vocab) - 1)

    model = build(options, fields_tuples, loss_weights)
    load_state(path, model)
    return model


def save(path, model):
    model_path = Path(path, constants.MODEL)
    model.save(model_path)
