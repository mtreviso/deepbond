import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from deepbond import constants
from deepbond.initialization import init_xavier, init_kaiming
from deepbond.models.model import Model


class RCNN(Model):
    """Recurrent Convolutional Neural Network.
    As described in: https://arxiv.org/pdf/1610.00211.pdf
    """

    def __init__(self, words_field, tags_field, options):
        super().__init__(words_field, tags_field)

        #
        # Embeddings
        #
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

        if options.freeze_embeddings:
            self.word_emb.weight.requires_grad = False

        features_size = options.word_embeddings_size

        #
        # CNN 1D
        #
        self.cnn_1d = nn.Conv1d(in_channels=features_size,
                                out_channels=options.conv_size,
                                kernel_size=options.kernel_size,
                                padding=options.kernel_size // 2)
        self.max_pool = nn.MaxPool1d(options.pool_length,
                                     padding=options.pool_length // 2)
        self.relu = torch.nn.ReLU()

        features_size = (options.conv_size // options.pool_length +
                         options.pool_length // 2)

        #
        # RNN
        #
        self.is_bidir = options.bidirectional
        self.sum_bidir = options.sum_bidir
        self.rnn_type = options.rnn_type

        if self.rnn_type == 'gru':
            rnn_class = nn.GRU
        elif self.rnn_type == 'lstm':
            rnn_class = nn.LSTM
        else:
            rnn_class = nn.RNN

        hidden_size = options.hidden_size[0]
        self.rnn = rnn_class(features_size,
                             hidden_size,
                             bidirectional=self.is_bidir,
                             batch_first=True)
        self.dropout_rnn = nn.Dropout(options.dropout)
        self.sigmoid = torch.nn.Sigmoid()

        features_size = hidden_size

        #
        # Linear
        #
        n = 1 if not self.is_bidir or self.sum_bidir else 2
        self.linear_out = nn.Linear(n * features_size, self.nb_classes)

        self.init_weights()
        self.is_built = True

    def init_weights(self):
        if self.cnn_1d is not None:
            init_kaiming(self.cnn_1d, dist='uniform', nonlinearity='relu')
        if self.rnn is not None:
            init_xavier(self.rnn, dist='uniform')
        if self.linear_out is not None:
            init_xavier(self.linear_out, dist='uniform')

    def forward(self, batch):
        assert self.is_built
        assert self._loss is not None

        h = batch.words
        mask = h != constants.PAD_ID
        lengths = mask.int().sum(dim=-1)

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
        h = pack(h, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(h)
        h, _ = unpack(h, batch_first=True)

        # if you'd like to sum instead of concatenate:
        if self.sum_bidir:
            h = (h[:, :, :self.rnn.hidden_size] +
                 h[:, :, self.rnn.hidden_size:])

        h = self.sigmoid(h)

        # apply dropout
        h = self.dropout_rnn(h)

        # if self.attn is not None:
        #     # self attention
        #     h, _ = self.attn(h, h, h, mask=mask)

        # (bs, ts, hidden_size) -> (bs, ts, nb_classes)
        h = self.linear_out(h)

        # (bs, ts, nb_classes) -> (bs, ts, nb_classes) in simplex
        h = F.log_softmax(h, dim=-1)

        # remove <bos> and <eos> tokens
        # (bs, ts, nb_classes) -> (bs, ts-2, nb_classes)
        h = h[:, 1:-1, :]

        return h
