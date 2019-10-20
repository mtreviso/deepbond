from collections import defaultdict
from pathlib import Path

from deepbond import constants
from deepbond import opts
from deepbond.models.cnn import CNN
from deepbond.models.cnn_attn import CNNAttention
from deepbond.models.cnn_attn_crf import CNNAttentionCRF
from deepbond.models.cnn_crf import CNNCRF
from deepbond.models.crf import LinearCRF
from deepbond.models.rcnn import RCNN
from deepbond.models.rcnn_attn import RCNNAttention
from deepbond.models.rcnn_attn_crf import RCNNAttentionCRF
from deepbond.models.rcnn_crf import RCNNCRF
from deepbond.models.rnn import RNN
from deepbond.models.rnn_attn_crf import RNNAttentionCRF
from deepbond.models.rnn_crf import RNNCRF
from deepbond.models.self_attn import SelfAttention
from deepbond.models.self_attn_crf import SelfAttentionCRF
from deepbond.models.transformer_attn import TransformerAttention

available_models = {
    'cnn': CNN,
    'cnn_attn': CNNAttention,
    'cnn_attn_crf': CNNAttentionCRF,
    'cnn_crf': CNNCRF,
    'crf': LinearCRF,
    'rcnn': RCNN,
    'rcnn_attn': RCNNAttention,
    'rcnn_attn_crf': RCNNAttentionCRF,
    'rcnn_crf': RCNNCRF,
    'rnn': RNN,
    'rnn_attn_crf': RNNAttentionCRF,
    'rnn_crf': RNNCRF,
    'self_attn': SelfAttention,
    'self_attn_crf': SelfAttentionCRF,
    'transformer_attn': TransformerAttention
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
