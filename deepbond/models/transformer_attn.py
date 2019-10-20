import torch
import torch.nn as nn

from deepbond import constants
from deepbond.initialization import init_xavier
from deepbond.models.model import Model
from deepbond.models.utils import neighbours_mask
from deepbond.modules.attention import Attention
from deepbond.modules.multi_headed_attention import MultiHeadedAttention
from deepbond.modules.scorer import (DotProductScorer, GeneralScorer,
                                     OperationScorer, MLPScorer)


class TransformerAttention(Model):
    """Transformer attention + linear projection"""

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
        self.attn_layer = nn.TransformerEncoderLayer(
            d_model=features_size,
            nhead=options.attn_nb_heads,
            dim_feedforward=options.attn_hidden_size,
            dropout=options.attn_dropout,
            activation='relu'
        )
        self.attn = nn.TransformerEncoder(
            self.attn_layer,
            num_layers=options.transformer_encoder_layers
        )

        #
        # Linear
        #
        self.linear_out = nn.Linear(features_size, self.nb_classes)

        self.init_weights()
        self.is_built = True

    def init_weights(self):
        if self.linear_out is not None:
            init_xavier(self.linear_out, dist='uniform')

    def forward(self, batch):
        assert self.is_built
        assert self._loss is not None

        h = batch.words
        mask = h != constants.PAD_ID

        # (bs, ts) -> (bs, ts, emb_dim)
        h = self.word_emb(h)
        h = self.dropout_emb(h)

        # (bs, ts, emb_dim) -> (bs, ts, emb_dim)
        # mask_key = neighbours_mask(h.shape[1], window_size=3)
        # mask_key = mask_key.to(h.device).unsqueeze(0).bool()
        # mask_key[mask_key == 1] = float("-inf")
        mask[mask == 1] = float("-inf")
        h = h.transpose(0, 1)
        h = self.attn(h, mask=mask, src_key_padding_mask=None)
        h = h.transpose(0, 1)

        # (bs, ts, emb_dim) -> (bs, ts, nb_classes)
        h = self.linear_out(h)

        # (bs, ts, nb_classes) -> (bs, ts, nb_classes) in simplex
        h = torch.log_softmax(h, dim=-1)

        # remove <bos> and <eos> tokens
        # (bs, ts, nb_classes) -> (bs, ts-2, nb_classes)
        h = h[:, 1:-1, :]

        return h
