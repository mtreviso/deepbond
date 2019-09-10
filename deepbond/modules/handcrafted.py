import torch
import torch.nn as nn


from deeptagger import constants


class HandCrafted(nn.Module):
    """Receives the input and calculate handcrafted features like prefixes,
    suffixes and capitalization."""

    def __init__(
        self,
        prefixes_field=None,
        suffixes_field=None,
        caps_field=None
    ):
        super().__init__()
        # layers
        self.prefixes_field = prefixes_field
        self.suffixes_field = suffixes_field
        self.caps_field = caps_field
        self.prefixes_emb = None
        self.prefix_length = None
        self.suffixes_emb = None
        self.suffix_length = None
        self.caps_emb = None
        self.caps_length = None
        self.is_built = False
        self.features_size = 0

    def build(self, options):
        self.features_size = 0
        if self.prefixes_field is not None:
            self.prefixes_emb = nn.Embedding(
                num_embeddings=len(self.prefixes_field.vocab),
                embedding_dim=options.prefix_embeddings_size,
                padding_idx=constants.PAD_ID,
            )
            self.prefix_length = (
                options.prefix_max_length - options.prefix_min_length + 1
            )
            self.features_size += (
                self.prefix_length * options.prefix_embeddings_size
            )

        if self.suffixes_field is not None:
            self.suffixes_emb = nn.Embedding(
                num_embeddings=len(self.suffixes_field.vocab),
                embedding_dim=options.suffix_embeddings_size,
                padding_idx=constants.PAD_ID,
            )
            self.suffix_length = (
                options.suffix_max_length - options.suffix_min_length + 1
            )
            self.features_size += (
                self.suffix_length * options.suffix_embeddings_size
            )

        if self.caps_field is not None:
            self.caps_emb = nn.Embedding(
                num_embeddings=len(self.caps_field.vocab),
                embedding_dim=options.caps_embeddings_size,
                padding_idx=constants.PAD_ID,
            )
            self.caps_length = 1
            self.features_size += options.caps_embeddings_size

        if options.freeze_embeddings:
            if self.prefixes_field is not None:
                self.prefixes_emb.weight.requires_grad = False
                self.prefixes_emb.bias.requires_grad = False
            if self.suffixes_field is not None:
                self.suffixes_emb.weight.requires_grad = False
                self.suffixes_emb.bias.requires_grad = False
            if self.caps_field is not None:
                self.caps_emb.weight.requires_grad = False
                self.caps_emb.bias.requires_grad = False

        self.is_built = True

    def init_weights(self):
        pass

    def forward(self, batch):
        assert self.is_built

        # (ts, bs) -> (bs, ts)
        bs, ts = batch.words.shape

        feats = []
        if self.prefixes_field is not None:
            # (bs, (ts-2)*(maxlen-minlen+1)) ->
            # (bs, (ts-2)*(max-min+1), emb_dim)
            h_pre = self.prefixes_emb(batch.prefixes)
            # (bs, (ts-2)*(max-min+1), emb_dim) ->
            # (bs, ts*(max-min+1), emb_dim)
            z = torch.zeros(h_pre.shape[0], self.prefix_length, h_pre.shape[2])
            h_pre = torch.cat((z, h_pre, z), dim=1)
            # (bs, ts*(max-min+1), emb_dim) -> (bs, ts, emb_dim*(max-min+1))
            h_pre = h_pre.view(bs, ts, -1)
            feats.append(h_pre)

        if self.suffixes_field is not None:
            h_suf = self.suffixes_emb(batch.suffixes)
            z = torch.zeros(h_suf.shape[0], self.suffix_length, h_suf.shape[2])
            h_suf = torch.cat((z, h_suf, z), dim=1)
            h_suf = h_suf.view(bs, ts, -1)
            feats.append(h_suf)

        if self.caps_field is not None:
            h_cap = self.caps_emb(batch.caps)
            z = torch.zeros(h_cap.shape[0], self.caps_length, h_cap.shape[2])
            h_cap = torch.cat((z, h_cap, z), dim=1)
            feats.append(h_cap)

        h = None
        if feats:
            h = torch.cat(feats, dim=-1)

        return h
