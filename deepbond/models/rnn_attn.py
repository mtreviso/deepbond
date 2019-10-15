import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from deepbond import constants
from deepbond.initialization import init_xavier
from deepbond.models.model import Model
from deepbond.modules.attention import Attention
from deepbond.modules.multi_headed_attention import MultiHeadedAttention
from deepbond.modules.scorer import (DotProductScorer, GeneralScorer,
                                     OperationScorer, MLPScorer)


class RNNAttention(Model):
    """Just a regular rnn (RNN, LSTM or GRU) network + Attention"""

    def __init__(self, words_field, tags_field, options):
        super().__init__(words_field, tags_field)

        word_embeddings = None
        if self.words_field.vocab.vectors is not None:
            word_embeddings = self.words_field.vocab.vectors
            options.word_embeddings_size = word_embeddings.size(1)

        self.word_emb = nn.Embedding(
            num_embeddings=len(self.words_field.vocab),
            embedding_dim=options.word_embeddings_size,
            padding_idx=constants.PAD_ID,
            _weight=word_embeddings
        )

        features_size = options.word_embeddings_size

        if options.freeze_embeddings:
            self.word_emb.weight.requires_grad = False

        self.is_bidir = options.bidirectional
        self.sum_bidir = options.sum_bidir
        self.rnn_type = options.rnn_type

        rnn_class = nn.RNN
        if self.rnn_type == 'gru':
            rnn_class = nn.GRU
        elif self.rnn_type == 'lstm':
            rnn_class = nn.LSTM

        hidden_size = options.hidden_size[0]
        self.hidden = None
        self.rnn = rnn_class(features_size,
                             hidden_size,
                             bidirectional=self.is_bidir,
                             batch_first=True)

        #
        # Attention
        #

        # they are equal for self-attention
        n = 1 if not self.is_bidir or self.sum_bidir else 2
        query_size = key_size = value_size = n * features_size

        if options.attn_scorer == 'dot_product':
            self.attn_scorer = DotProductScorer(scaled=True)
        elif options.attn_scorer == 'general':
            self.attn_scorer = GeneralScorer(query_size, key_size)
        elif options.attn_scorer == 'add':
            self.attn_scorer = OperationScorer(query_size, key_size,
                                               options.attn_hidden_size,
                                               op='add')
        elif options.attn_scorer == 'concat':
            self.attn_scorer = OperationScorer(query_size, key_size,
                                               options.attn_hidden_size,
                                               op='concat')
        elif options.attn_scorer == 'mlp':
            self.attn_scorer = MLPScorer(query_size, key_size)
        else:
            raise Exception('Attention scorer `{}` not available'.format(
                options.attn_scorer))

        if options.attn_type == 'regular':
            self.attn = Attention(self.attn_scorer,
                                  dropout=options.attn_dropout)
        elif options.attn_type == 'multihead':
            self.attn = MultiHeadedAttention(
                self.attn_scorer,
                options.attn_nb_heads,
                query_size,
                key_size,
                value_size,
                options.attn_multihead_hidden_size,
                dropout=options.attn_dropout
            )
            features_size = options.attn_multihead_hidden_size
        else:
            raise Exception('Attention `{}` not available'.format(
                options.attn_type))


        #
        # Linear
        #
        self.linear_out = nn.Linear(features_size, self.nb_classes)

        self.selu = torch.nn.SELU()
        self.dropout_emb = nn.Dropout(options.emb_dropout)
        self.dropout_rnn = nn.Dropout(options.rnn_dropout)

        self.init_weights()
        self.is_built = True

    def init_weights(self):
        if self.rnn is not None:
            init_xavier(self.rnn, dist='uniform')
        if self.linear_out is not None:
            init_xavier(self.linear_out, dist='uniform')

    def init_hidden(self, batch_size, hidden_size, device=None):
        # The axes semantics are (nb_layers, minibatch_size, hidden_dim)
        nb_layers = 2 if self.is_bidir else 1
        if self.rnn_type == 'lstm':
            return (torch.zeros(nb_layers, batch_size, hidden_size).to(device),
                    torch.zeros(nb_layers, batch_size, hidden_size).to(device))
        else:
            return torch.zeros(nb_layers, batch_size, hidden_size).to(device)

    def forward(self, batch):
        assert self.is_built
        assert self._loss is not None

        batch_size = batch.words.shape[0]
        device = batch.words.device

        # (ts, bs) -> (bs, ts)
        h = batch.words
        mask = h != constants.PAD_ID
        lengths = mask.int().sum(dim=-1)

        # initialize RNN hidden state
        self.hidden = self.init_hidden(
            batch_size, self.rnn.hidden_size, device=device
        )

        # (bs, ts) -> (bs, ts, emb_dim)
        h = self.word_emb(h)
        h = self.dropout_emb(h)

        # (bs, ts, pool_size) -> (bs, ts, hidden_size)
        h = pack(h, lengths, batch_first=True, enforce_sorted=False)
        h, self.hidden = self.rnn(h, self.hidden)
        h, _ = unpack(h, batch_first=True)

        # if you'd like to sum instead of concatenate:
        if self.sum_bidir:
            h = (h[:, :, :self.rnn.hidden_size] +
                 h[:, :, self.rnn.hidden_size:])

        h = self.selu(h)
        h = self.dropout_rnn(h)

        # (bs, ts, hidden_size) -> (bs, ts, hidden_size)
        h, _ = self.attn(h, h, h, mask=mask)

        # (bs, ts, hidden_size) -> (bs, ts, nb_classes)
        h = self.linear_out(h)

        # (bs, ts, nb_classes) -> (bs, ts, nb_classes) in simplex
        h = torch.log_softmax(h, dim=-1)

        # remove <bos> and <eos> tokens
        # (bs, ts, nb_classes) -> (bs, ts-2, nb_classes)
        h = h[:, 1:-1, :]

        return h
