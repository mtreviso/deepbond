import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from deepbond import constants
from deepbond.models.model import Model
from deepbond.modules.attention import Attention
from deepbond.modules.scorer import DotProductScorer


class RCNN(Model):
    """Recurrent Convolutional Neural Network.
    As described in: https://arxiv.org/pdf/1610.00211.pdf
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # layers
        self.word_emb = None
        self.dropout_emb = None
        self.cnn_1d = None
        self.max_pool = None
        self.is_bidir = None
        self.sum_bidir = None
        self.rnn_type = None
        self.rnn = None
        self.hidden = None
        self.dropout_rnn = None
        self.linear_out = None
        self.relu = None
        self.sigmoid = None

    def build(self, options, loss_weights=None):
        # prefix_embeddings_size = options.prefix_embeddings_size
        # suffix_embeddings_size = options.suffix_embeddings_size
        # caps_embeddings_size = options.caps_embeddings_size
        hidden_size = options.hidden_size[0]
        if loss_weights is not None:
            loss_weights = torch.tensor(loss_weights).float()

        word_embeddings = None
        if self.words_field.vocab.vectors is not None:
            word_embeddings = self.words_field.vocab.vectors
            options.word_embeddings_size = word_embeddings.size(1)

        self.word_emb = nn.Embedding(
            num_embeddings=len(self.words_field.vocab),
            embedding_dim=options.word_embeddings_size,
            padding_idx=constants.PAD_ID,
            _weight=word_embeddings,
        )
        self.dropout_emb = nn.Dropout(options.emb_dropout)

        features_size = options.word_embeddings_size
        if options.freeze_embeddings:
            self.word_emb.weight.requires_grad = False
            self.word_emb.bias.requires_grad = False

        self.cnn_1d = nn.Conv1d(in_channels=features_size,
                                out_channels=options.conv_size,
                                kernel_size=options.kernel_size,
                                padding=options.kernel_size // 2)

        self.max_pool = nn.MaxPool1d(options.pool_length,
                                     padding=options.pool_length // 2)

        self.is_bidir = options.bidirectional
        self.sum_bidir = options.sum_bidir
        self.rnn_type = options.rnn_type

        rnn_class = nn.RNN
        if self.rnn_type == 'gru':
            rnn_class = nn.GRU
        elif self.rnn_type == 'lstm':
            rnn_class = nn.LSTM
        self.rnn = rnn_class(options.conv_size // options.pool_length +
                             options.pool_length // 2,
                             hidden_size,
                             bidirectional=self.is_bidir,
                             batch_first=True)
        self.hidden = None
        self.dropout_rnn = nn.Dropout(options.dropout)

        n = 2 if self.is_bidir else 1
        n = 1 if self.sum_bidir else n
        self.linear_out = nn.Linear(n * hidden_size, self.nb_classes)

        self.scorer = DotProductScorer(scaled=True)
        self.attn = Attention(self.scorer)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.init_weights()

        # Loss
        self._loss = nn.NLLLoss(weight=loss_weights,
                                ignore_index=constants.TAGS_PAD_ID)
        self.is_built = True

    def init_weights(self):
        def init_xavier(module):
            for name, param in module.named_parameters():
                if 'bias' in name:
                    torch.nn.init.constant_(param, 0.)
                elif 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
        init_xavier(self.rnn)
        init_xavier(self.cnn_1d)
        init_xavier(self.linear_out)

    def init_hidden(self, batch_size, hidden_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        num_layers = 2 if self.is_bidir else 1
        if self.rnn_type == 'lstm':
            return (torch.zeros(num_layers, batch_size, hidden_size),
                    torch.zeros(num_layers, batch_size, hidden_size))
        else:
            return torch.zeros(num_layers, batch_size, hidden_size)

    def forward(self, batch):
        assert self.is_built

        h = batch.words
        mask = h != constants.PAD_ID
        lengths = mask.int().sum(dim=-1)

        # initialize RNN hidden state
        self.hidden = self.init_hidden(h.shape[0], self.rnn.hidden_size)

        # (bs, ts) -> (bs, ts, emb_dim)
        h = self.word_emb(h)
        h = self.dropout_emb(h)

        # Turn (bs, ts, emb_dim) into (bs, emb_dim, ts) for CNN
        h = h.transpose(1, 2)

        # (bs, emb_dim, ts) -> (bs, conv_size, ts)
        h = self.relu(self.cnn_1d(h))

        # Turn (bs, conv_size, ts) into (bs, ts, conv_size) for Pooling
        h = h.transpose(1, 2)

        # (bs, ts, conv_size) -> (bs, ts, pool_size)
        h = self.max_pool(h)

        # (bs, ts, pool_size) -> (bs, ts, hidden_size)
        h = pack(h, lengths, batch_first=True)
        h, self.hidden = self.rnn(h, self.hidden)
        h, _ = unpack(h, batch_first=True)
        h = self.dropout_rnn(h)

        # if you'd like to sum instead of concatenate:
        if self.sum_bidir:
            h = (h[:, :, :self.rnn.hidden_size] +
                 h[:, :, self.rnn.hidden_size:])

        # self attention
        h, _ = self.attn(h, h, h, mask=mask)

        # (bs, ts, hidden_size) -> (bs, ts, nb_classes)
        h = self.linear_out(h)

        # (bs, ts, nb_classes) -> (bs, ts, nb_classes) in simplex
        h = F.log_softmax(h, dim=-1)

        # remove <bos> and <eos> tokens
        # (bs, ts, nb_classes) -> (bs, ts-2, nb_classes)
        h = h[:, 1:-1, :]

        return h
