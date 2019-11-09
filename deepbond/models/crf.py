import torch
import torch.nn as nn

from deepbond import constants
from deepbond.initialization import init_xavier
from deepbond.models.model import Model
from deepbond.modules.crf import CRF


class LinearCRF(Model):
    """Just a linear layer followed by a CRF"""

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

        # Hidden
        self.linear_hidden = None
        self.sigmoid = nn.Sigmoid()
        hidden_size = options.hidden_size[0]
        if hidden_size > 0:
            self.linear_hidden = nn.Linear(features_size, hidden_size)
            features_size = hidden_size

        #
        # Linear
        #
        self.linear_out = nn.Linear(features_size, self.nb_classes)

        self.crf = CRF(
            self.nb_classes,
            bos_tag_id=self.tags_field.vocab.stoi['_'],  # hack
            eos_tag_id=self.tags_field.vocab.stoi['.'],  # hack
            pad_tag_id=None,
            batch_first=True,
        )
        # self.crf.apply_pad_constraints()

        self.init_weights()
        self.is_built = True

    def init_weights(self):
        if self.linear_out is not None:
            init_xavier(self.linear_out, dist='uniform')

    def build_loss(self, loss_weights=None):
        self._loss = self.crf

    def loss(self, emissions, gold):
        mask = gold != constants.TAGS_PAD_ID
        crf_gold = gold.clone()
        # it can be any valid tag id number, since they will be masked out in
        # the CRF anyway. Here I choose 0 (can't be pad_id because num_tags is
        # len(tags_vocab) -1), so there is no transition to pad, unless
        # we emit a score for pad as well, which can make the neural net to
        # think that pad is a valid label
        crf_gold[mask == 0] = 0
        return self._loss(emissions, crf_gold, mask=mask.float())

    def predict_classes(self, batch):
        emissions = self.forward(batch)
        mask = batch.words != constants.PAD_ID
        _, path = self.crf.decode(emissions, mask=mask[:, 2:].float())
        return [torch.tensor(p) for p in path]

    def predict_proba(self, batch):
        raise Exception('Predict() probability is not available.')

    def forward(self, batch):
        assert self.is_built
        assert self._loss is not None

        h = batch.words
        # mask = h != constants.PAD_ID

        # (bs, ts) -> (bs, ts, emb_dim)
        h = self.word_emb(h)
        h = self.dropout_emb(h)

        if self.linear_hidden is not None:
            h = self.linear_hidden(h)
            h = self.sigmoid(h)

        # (bs, ts, emb_dim) -> (bs, ts, nb_classes)
        h = self.linear_out(h)

        # remove <bos> and <eos> tokens
        # (bs, ts, nb_classes) -> (bs, ts-2, nb_classes)
        h = h[:, 1:-1, :]

        return h
