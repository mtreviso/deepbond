import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from deepbond import constants
from deepbond.initialization import init_xavier
from deepbond.models.model import Model
from deepbond.modules.crf import CRF


class RNNCRF(Model):
    """Just a regular rnn (RNN, LSTM or GRU) network + CRF."""

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

        n = 1 if not self.is_bidir or self.sum_bidir else 2
        self.linear_out = nn.Linear(n * hidden_size, self.nb_classes)

        self.crf = CRF(
            self.nb_classes,
            bos_tag_id=self.tags_field.vocab.stoi['_'],  # hack
            eos_tag_id=self.tags_field.vocab.stoi['.'],  # hack
            pad_tag_id=self.tags_field.vocab.stoi[constants.PAD],
            batch_first=True,
        )
        self.crf.apply_pad_constraints()

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

    @property
    def nb_classes(self):
        return len(self.tags_field.vocab.stoi)  # include pad index

    def build_loss(self, loss_weights=None):
        self._loss = self.crf

    def loss(self, emissions, gold):
        mask = gold != constants.TAGS_PAD_ID
        return self._loss(emissions, gold, mask=mask.float())

    def predict_classes(self, batch):
        emissions = self.forward(batch)
        mask = batch.words != constants.PAD_ID
        _, path = self.crf.decode(emissions, mask=mask[:, 2:].float())
        return [torch.tensor(p) for p in path]

    def predict_proba(self, batch):
        raise Exception('Predict() probability is not available.')

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

        # (bs, ts, hidden_size) -> (bs, ts, nb_classes)
        h = self.linear_out(h)

        # remove <bos> and <eos> tokens
        # (bs, ts, nb_classes) -> (bs, ts-2, nb_classes)
        h = h[:, 1:-1, :]

        return h
