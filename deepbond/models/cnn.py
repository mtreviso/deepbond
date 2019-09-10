import torch
import torch.nn as nn
import torch.nn.functional as F

from deeptagger import constants
from deeptagger.models.model import Model


class CNN(Model):
    """Simple Convolutional Neural Network 1D."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # layers
        self.word_emb = None
        self.dropout_emb = None
        self.cnn_1d = None
        self.max_pool = None
        self.linear_out = None
        self.relu = None
        self.sigmoid = None

    def build(self, options):
        loss_weights = None
        if options.loss_weights == 'balanced':
            # TODO
            # loss_weights = calc_balanced(loss_weights, tags_field)
            loss_weights = torch.FloatTensor(loss_weights)

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

        if self.use_handcrafed:
            self.handcrafted.build(options)
            features_size += self.handcrafted.features_size

        self.cnn_1d = nn.Conv1d(in_channels=features_size,
                                out_channels=options.conv_size,
                                kernel_size=options.kernel_size,
                                padding=options.kernel_size // 2)
        self.max_pool = nn.MaxPool1d(options.pool_length,
                                     padding=options.pool_length // 2)
        self.linear_out = nn.Linear(options.conv_size // options.pool_length +
                                    options.pool_length // 2,
                                    self.nb_classes)
        self.relu = torch.nn.ReLU()

        self.init_weights()
        self._loss = nn.NLLLoss(weight=loss_weights,
                                ignore_index=constants.TAGS_PAD_ID)
        self.is_built = True

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.cnn_1d.weight)
        torch.nn.init.constant_(self.cnn_1d.bias, 0.)
        torch.nn.init.xavier_uniform_(self.linear_out.weight)
        torch.nn.init.constant_(self.linear_out.bias, 0.)

    def forward(self, batch):
        assert self.is_built

        h = batch.words

        # (bs, ts) -> (bs, ts, emb_dim)
        h = self.word_emb(h)

        feats = [h]
        if self.use_handcrafed:
            feats.append(self.handcrafted.forward(batch))

        if feats:
            h = torch.cat(feats, dim=-1)

        h = self.dropout_emb(h)

        # Turn (bs, ts, emb_dim) into (bs, emb_dim, ts) for CNN
        h = h.transpose(1, 2)

        # (bs, emb_dim, ts) -> (bs, conv_size, ts)
        h = self.relu(self.cnn_1d(h))

        # Turn (bs, conv_size, ts) into (bs, ts, conv_size) for Pooling
        h = h.transpose(1, 2)

        # (bs, ts, conv_size) -> (bs, ts, pool_size)
        h = self.max_pool(h)

        # (bs, ts, pool_size) -> (bs, ts, nb_classes)
        h = F.log_softmax(self.linear_out(h), dim=-1)

        # remove <bos> and <eos> tokens
        # (bs, ts, nb_classes) -> (bs, ts-2, nb_classes)
        h = h[:, 1:-1, :]

        return h
