# flake8: noqa: E501

from torch import nn
from torch.nn import functional as F

from deeptagger.modules.layer_norm import LayerNorm
from deeptagger.modules.multi_headed_attention import MultiHeadedAttention
from deeptagger.modules.pointwise_ffn import PositionwiseFeedForward
from deeptagger.modules.positional_embedding import PositionalEmbedding
from deeptagger.modules.scorer import DotProductScorer
from deeptagger.models.utils import clones


class TransformerAttention(nn.Module):
    """ Simple wrapper to append projection, dropout and layer norm
    to the MultiHeadedAttention module."""

    def __init__(self, attn, dropout=0.0):
        super().__init__()
        self.attn = attn
        self.proj = nn.Linear(attn.hidden_size, attn.value_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(attn.value_size)

    def forward(self, x, memory=None, mask=None):
        if memory is None:
            # self-attention
            hidden_x, _ = self.attn(x, x, x, mask=mask)
        else:
            # attention over source outputs
            hidden_x, _ = self.attn(x, memory, memory, mask=mask)
        hidden_x = self.proj(hidden_x)
        hidden_x = self.dropout(hidden_x)
        return self.norm(x + hidden_x)


class TransformerFFN(nn.Module):
    """ Simple wrapper to append dropout and layer norm
    to the PositionwiseFFN module."""

    def __init__(self, position_ffn, dropout=0.0):
        super().__init__()
        self.position_ffn = position_ffn
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(position_ffn.hidden_size)

    def forward(self, x):
        hidden_x = self.position_ffn(x)
        hidden_x = self.dropout(hidden_x)
        return self.norm(x + hidden_x)


class EncoderLayer(nn.Module):
    """Implements a single layer for the Encoder block:

    This implementation is based on BERT implementation, which
    is slightly different from The Annotated Transformer, but is
    faithful to the original AIAYN paper:

    x = Norm(x + Dropout(Projection(SelfMultiHeadedAttention(x))))
    x = Norm(x + Dropout(Projection(SourceMultiHeadedAttention(x, m))))

    Args:
        attn (nn.Module): attention mechanism object
        position_ffn (nn.Module): position-wise ffn object
        dropout (float): dropout rate (default: 0.)
    """

    def __init__(self, attn, position_ffn, dropout=0.0):
        super().__init__()
        self.attn = TransformerAttention(attn, dropout=dropout)
        self.position_ffn = TransformerFFN(position_ffn, dropout=dropout)

    def forward(self, x, mask):
        x = self.attn(x, mask=mask)
        x = self.position_ffn(x)
        return x


class DecoderLayer(nn.Module):
    """Implements a single layer for the Decoder block:

    This implementation is based on BERT implementation, which
    is slightly different from The Annotated Transformer, but is
    faithful to the original AIAYN paper:

    x = Norm(x + Dropout(Projection(MultiHeadedAttention(x))))
    x = Norm(x + Dropout(Projection(MultiHeadedAttention(x))))
    out = Norm(x + Dropout(PositionwiseFeedForward(x)))

    Args:
        self_attn (nn.Module): attention mechanism object for self attention
        source_attn (nn.Module): attention mechanism object for attention
            over source outputs
        position_ffn (nn.Module): position-wise ffn object
        dropout (float): dropout rate (default: 0.)
    """

    def __init__(self, self_attn, src_attn, position_ffn, dropout=0.0):
        super().__init__()
        self.self_attn = TransformerAttention(self_attn, dropout=dropout)
        self.src_attn = TransformerAttention(src_attn, dropout=dropout)
        self.position_ffn = TransformerFFN(position_ffn, dropout=dropout)

    def forward(self, x, tgt_mask, memory, src_mask):
        x = self.self_attn(x, mask=tgt_mask)
        x = self.src_attn(x, memory=memory, mask=src_mask)
        x = self.position_ffn(x)
        return x


class TransformerEncoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, nb_layers=1):
        super().__init__()
        self.layers = clones(layer, nb_layers)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerDecoder(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer, nb_layers=1):
        super().__init__()
        self.layers = clones(layer, nb_layers)

    def forward(self, x, tgt_mask, memory, src_mask):
        for layer in self.layers:
            x = layer(x, tgt_mask, memory, src_mask)
        return x


class TransformerGenerator(nn.Module):
    """Implements a standard linear + softmax generation step."""

    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.proj(x)
        x = F.log_softmax(x, dim=-1)
        return x


class Transformer(nn.Module):
    """Make an encoder-decoder transformer model."""

    def __init__(
        self,
        source_vocab_size,
        target_vocab_size,
        nb_layers=6,
        hidden_size=512,
        attn_hidden_size=256,
        ff_hidden_size=2048,
        nb_heads=8,
        max_seq_len=5000,
        dropout_encoder=0.1,
        dropout_decoder=0.1,
        dropout_attention=0.1,
        dropout_emb=0.1,
    ):
        super().__init__()

        # for dot product they should have the same hidden size
        query_size = key_size = value_size = hidden_size

        # encoder layer blocks
        encoder_scorer = DotProductScorer()
        encoder_attn = MultiHeadedAttention(
            encoder_scorer,
            nb_heads,
            query_size,
            key_size,
            value_size,
            attn_hidden_size,
            dropout=dropout_attention,
        )
        encoder_ff = PositionwiseFeedForward(hidden_size, ff_hidden_size)
        encoder_layer = EncoderLayer(
            encoder_attn,
            encoder_ff,
            dropout=dropout_encoder
        )

        # decoder layer blocks
        decoder_self_scorer = DotProductScorer()
        decoder_self_attn = MultiHeadedAttention(
            decoder_self_scorer,
            nb_heads,
            query_size,
            key_size,
            value_size,
            attn_hidden_size,
            dropout=dropout_attention,
        )
        decoder_source_scorer = DotProductScorer()
        decoder_source_attn = MultiHeadedAttention(
            decoder_source_scorer,
            nb_heads,
            query_size,
            key_size,
            value_size,
            attn_hidden_size,
            dropout=dropout_attention,
        )
        decoder_ff = PositionwiseFeedForward(hidden_size, ff_hidden_size)
        decoder_layer = DecoderLayer(
            decoder_self_attn,
            decoder_source_attn,
            decoder_ff,
            dropout=dropout_decoder,
        )

        self.encoder_emb = PositionalEmbedding(
            source_vocab_size,
            hidden_size,
            max_seq_len=max_seq_len,
            dropout=dropout_emb,
        )
        self.decoder_emb = PositionalEmbedding(
            target_vocab_size,
            hidden_size,
            max_seq_len=max_seq_len,
            dropout=dropout_emb,
        )
        self.encoder = TransformerEncoder(encoder_layer, nb_layers=nb_layers)
        self.decoder = TransformerDecoder(decoder_layer, nb_layers=nb_layers)
        self.generator = TransformerGenerator(hidden_size, target_vocab_size)

        self._init_params()

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        src_emb = self.encoder_emb(src)
        return self.encoder(src_emb, src_mask)

    def decode(self, tgt, tgt_mask, memory, src_mask):
        tgt_emb = self.decoder_emb(tgt)
        return self.decoder(tgt_emb, tgt_mask, memory, src_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        return self.decode(tgt, tgt_mask, memory, src_mask)

    def generate(self, src, tgt, src_mask, tgt_mask):
        x = self.forward(src, tgt, src_mask, tgt_mask)
        return self.generator(x)


class SequentialTransformerEncoder(Transformer):
    pass


if __name__ == '__main__':

    import torch
    from deeptagger.models.utils import sequence_mask, subsequent_mask

    torch.manual_seed(1)
    # torch.cuda.manual_seed(1)

    batch_size = 8
    source_len = 7
    target_len = 5
    source_vocab_size = 10
    target_vocab_size = 5

    source = torch.randint(0, source_vocab_size, size=(batch_size, source_len)).long()
    target = torch.randint(0, target_vocab_size, size=(batch_size, target_len)).long()

    source_mask = sequence_mask(torch.LongTensor([5, 3, 7, 4, 5, 4, 3, 6]))

    target_mask_pad = sequence_mask(torch.LongTensor([5, 1, 2, 4, 3, 5, 5, 2]))
    target_mask_valid = subsequent_mask(target_len)

    # broadcast at timestep dim is infered automatically by attention module
    # or you can set manually by: source_mask.unsqueeze(-2)
    source_mask = source_mask
    target_mask = target_mask_pad.unsqueeze(-2) & target_mask_valid.unsqueeze(0)

    print(source_mask.shape, target_mask.shape)
    print(source_mask[1])
    print(target_mask[2])

    model = Transformer(source_vocab_size,
                        target_vocab_size,
                        hidden_size = 20,
                        nb_layers = 1,
                        ff_hidden_size = 12,
                        nb_heads = 2,
                        attn_hidden_size = 18,
                        max_seq_len = 100,
                        dropout_encoder = 0.1,
                        dropout_decoder = 0.1,
                        dropout_attention = 0.1,
                        dropout_emb = 0.1)

    with torch.no_grad():

        memory = model.encode(source, source_mask)
        print(memory.shape, memory.sum(), memory.mean(), memory.min(), memory.max())

        pred = model.decode(target, target_mask, memory, source_mask)
        print(pred.shape, pred.sum(), pred.mean(), pred.min(), pred.max())

        pred = model.generate(source, target, source_mask, target_mask)
        print(pred.shape, pred.sum(), pred.mean(), pred.min(), pred.max())

        pred = model(source, target, source_mask, target_mask)
        print(pred.shape, pred.sum(), pred.mean(), pred.min(), pred.max())

        pred = model.generator(pred)
        print(pred.shape, pred.sum(), pred.mean(), pred.min(), pred.max())
