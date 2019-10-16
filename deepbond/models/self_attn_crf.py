import torch
import torch.nn as nn

from deepbond import constants
from deepbond.initialization import init_xavier
from deepbond.models.model import Model
from deepbond.modules.attention import Attention
from deepbond.modules.crf import CRF
from deepbond.modules.multi_headed_attention import MultiHeadedAttention
from deepbond.modules.scorer import (DotProductScorer, GeneralScorer,
                                     OperationScorer, MLPScorer)


class SelfAttentionCRF(Model):
    """Self Attention + CRF on top"""

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
        # Attention
        #

        # they are equal for self-attention
        query_size = key_size = value_size = features_size

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

        self.crf = CRF(
            self.nb_classes,
            bos_tag_id=self.tags_field.vocab.stoi['_'],  # hack
            eos_tag_id=self.tags_field.vocab.stoi['.'],  # hack
            pad_tag_id=self.tags_field.vocab.stoi[constants.PAD],
            batch_first=True,
        )
        self.crf.apply_pad_constraints()

        self.init_weights()
        self.is_built = True

    def init_weights(self):
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

    def forward(self, batch):
        assert self.is_built
        assert self._loss is not None

        h = batch.words
        mask = h != constants.PAD_ID

        # (bs, ts) -> (bs, ts, emb_dim)
        h = self.word_emb(h)
        h = self.dropout_emb(h)

        # (bs, ts, emb_dim) -> (bs, ts, emb_dim)
        h, _ = self.attn(h, h, h, mask=mask)

        # (bs, ts, emb_dim) -> (bs, ts, nb_classes)
        h = self.linear_out(h)

        # remove <bos> and <eos> tokens
        # (bs, ts, nb_classes) -> (bs, ts-2, nb_classes)
        h = h[:, 1:-1, :]

        return h
