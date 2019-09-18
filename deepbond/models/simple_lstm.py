import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from deepbond import constants
from deepbond.models.model import Model


class SimpleLSTM(Model):
    """Just a regular one-layer LSTM network."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # layers
        self.word_emb = None
        self.dropout_emb = None
        self.is_bidir = None
        self.sum_bidir = None
        self.lstm = None
        self.hidden = None
        self.dropout_gru = None
        self.linear_out = None
        self.relu = None
        self.sigmoid = None

    def build(self, options, loss_weights=None):
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
            _weight=word_embeddings
        )

        features_size = options.word_embeddings_size

        if options.freeze_embeddings:
            self.word_emb.weight.requires_grad = False

        self.is_bidir = options.bidirectional
        self.sum_bidir = options.sum_bidir
        self.lstm = nn.LSTM(features_size,
                            hidden_size,
                            bidirectional=options.bidirectional,
                            batch_first=True)
        self.hidden = None

        n = 2 if self.is_bidir else 1
        n = 1 if self.sum_bidir else n
        self.linear_out = nn.Linear(n * hidden_size, self.nb_classes)

        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.dropout_emb = nn.Dropout(options.emb_dropout)
        self.dropout_gru = nn.Dropout(options.dropout)

        self.init_weights()

        # Loss
        self._loss = nn.NLLLoss(weight=loss_weights,
                                ignore_index=constants.TAGS_PAD_ID)

        self.is_built = True

    def init_weights(self):
        pass

    def init_hidden(self, batch_size, hidden_size, device=None):
        # The axes semantics are (nb_layers, minibatch_size, hidden_dim)
        nb_layers = 2 if self.is_bidir else 1
        return (torch.zeros(nb_layers, batch_size, hidden_size).to(device),
                torch.zeros(nb_layers, batch_size, hidden_size).to(device))

    def forward(self, batch):
        assert self.is_built

        batch_size = batch.words.shape[0]
        device = batch.words.device

        # (ts, bs) -> (bs, ts)
        h = batch.words
        mask = h != constants.PAD_ID
        lengths = mask.int().sum(dim=-1)

        # initialize GRU hidden state
        self.hidden = self.init_hidden(
            batch_size, self.lstm.hidden_size, device=device
        )

        # (bs, ts) -> (bs, ts, emb_dim)
        h = self.word_emb(h)
        h = self.dropout_emb(h)

        # (bs, ts, pool_size) -> (bs, ts, hidden_size)
        h = pack(h, lengths, batch_first=True, enforce_sorted=False)
        h, self.hidden = self.lstm(h, self.hidden)
        h, _ = unpack(h, batch_first=True)

        # if you'd like to sum instead of concatenate:
        if self.sum_bidir:
            h = (h[:, :, :self.lstm.hidden_size] +
                 h[:, :, self.lstm.hidden_size:])

        h = self.dropout_gru(h)

        # (bs, ts, hidden_size) -> (bs, ts, nb_classes)
        h = F.log_softmax(self.linear_out(h), dim=-1)

        # remove <bos> and <eos> tokens
        # (bs, ts, nb_classes) -> (bs, ts-2, nb_classes)
        h = h[:, 1:-1, :]

        return h
