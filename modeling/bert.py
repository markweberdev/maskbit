"""This file contains the definition of the Generator, which is based on Bert.

We thank the following public implementations for inspiring this code:
    https://github.com/google-research/magvit/blob/main/videogvt/models/simplified_bert.py
"""

import math
from typing import List, Tuple, Union, Optional

import torch
from einops import rearrange

from modeling.modules import BaseModel


class BertFeedForward(torch.nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0., use_prenorm: bool = False):
        """ Initialize the Multi-Layer Perceptron (MLP).

            Args:
                dim -> int: Dimension of the input tensor.
                hidden_dim -> int: Dimension of the hidden layer.
                dropout -> float: Dropout rate. Defaults to 0.
                use_prenorm -> bool: Flag setting prenorm or postnorm. Defaults to False.
        """
        super(BertFeedForward, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim, bias=True),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, dim, bias=True),
            torch.nn.Dropout(dropout),
        )
        self.norm = torch.nn.LayerNorm(dim, eps=1e-12)
        self.use_prenorm = use_prenorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the MLP module.

            Args:
                x -> torch.Tensor: Input tensor.
            Returns:
                torch.Tensor: Output of MLP layer.
        """
        if self.use_prenorm:
            return self._forward_prenorm(x)
        else:
            return self._forward_postnorm(x)

    def _forward_prenorm(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the MLP module with prenorm.

            Args:
                x -> torch.Tensor: Input tensor.
            Returns:
                torch.Tensor: Output of MLP layer.
        """
        y = self.norm(x)
        out = self.net(y)
        return out + x

    def _forward_postnorm(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the MLP module with postnorm.

            Args:
                x -> torch.Tensor: Input tensor.
            Returns:
                torch.Tensor: Output of MLP layer.
        """
        out = self.net(x)
        return self.norm(out + x)


class BertAttention(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0., use_prenorm: bool = False):
        """ Initialize the BertAttention module.

            Args:
                embed_dim -> int: Dimension of the input tensor.
                num_heads -> int: Number of heads in the multi-head attention.
                dropout -> float: Dropout rate. Defaults to 0.
                use_prenorm -> bool: Flag setting prenorm or postnorm. Defaults to False.
        """
        super(BertAttention, self).__init__()
        self.mha = torch.nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True, bias=True)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.norm = torch.nn.LayerNorm(embed_dim, eps=1e-12)
        self.use_prenorm = use_prenorm

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Forward pass through the BertAttention module.

            Args:
                x -> torch.Tensor: Input tensor.
                return_attn -> bool: Flag setting whether to compute attention weights or not.
                    Setting this to false can enable faster MHSA in pytorch 2.x. Defaults to False.
            Returns:
                (attention_value, attention_weight) -> Tuple[torch.Tensor, torch.Tensor]: 
                    First element is the output of this attention layer, while the second element contain 
                    the attention weights if computed.
        """
        if self.use_prenorm:
            return self._forward_prenorm(x, return_attn)
        else:
            return self._forward_postnorm(x, return_attn)

    def _forward_prenorm(self, x: torch.Tensor, return_attn: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Forward pass through the BertAttention module with Prenorm.

            Args:
                x -> torch.Tensor: Input tensor.
                return_attn -> bool: Flag setting whether to compute attention weights or not.
                    Setting this to false can enable faster MHSA in pytorch 2.x. Defaults to False.
            Returns:
                (attention_value, attention_weight) -> Tuple[torch.Tensor, torch.Tensor]: 
                    First element is the output of this attention layer, while the second element contain 
                    the attention weights if computed.
        """
        y = self.norm(x)
        attention_value, attention_weight = self.mha(y, y, y, need_weights=return_attn)
        attention_value = self.dropout(attention_value)
        out = attention_value + x

        return out, attention_weight

    def _forward_postnorm(self, x: torch.Tensor, return_attn: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Forward pass through the BertAttention module with Postnorm.

            Args:
                x -> torch.Tensor: Input tensor.
                return_attn -> bool: Flag setting whether to compute attention weights or not.
                    Setting this to false can enable faster MHSA in pytorch 2.x. Defaults to False.
            Returns:
                (attention_value, attention_weight) -> Tuple[torch.Tensor, torch.Tensor]: 
                    First element is the output of this attention layer, while the second element contain 
                    the attention weights if computed.
        """
        attention_value, attention_weight = self.mha(x, x, x, need_weights=return_attn)
        attention_value = self.dropout(attention_value)
        out = self.norm(attention_value + x)

        return out, attention_weight


class TransformerEncoder(torch.nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, mlp_dim: int, dropout: float = 0., use_prenorm: bool = False):
        """ Initialize the Transformer module.
        
            Args:
                dim -> int: Dimension of the input tensor.
                depth -> int: Number of attention layers.
                heads -> int: Number of attention heads.
                mlp_dim -> int: Dimension of the MLP.
                dropout -> float: Dropout rate. Defaults to 0.
                use_prenorm -> bool: Flag setting prenorm or postnorm. Defaults to False.
        """
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(torch.nn.ModuleList([
                BertAttention(dim, heads, dropout=dropout, use_prenorm=use_prenorm),
                BertFeedForward(dim, mlp_dim, dropout=dropout, use_prenorm=use_prenorm)
            ]))

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """ Forward pass through the Attention module.

            Args:
                x -> torch.Tensor: Input tensor.
                return_attn -> bool: Flag setting whether to compute attention weights or not.
                    Setting this to false can enable faster MHSA in pytorch 2.x. Defaults to False.
            Returns:
                (transformer_output, [attention_weights]) -> Tuple[torch.Tensor, List[torch.Tensor]]: 
                    First element is the output of this transformer, while the second element contain 
                    the attention weights if computed.
        """
        l_attn = []
        for attn, ffn in self.layers:
            x, attention_weight = attn(x, return_attn=return_attn)
            x = ffn(x)
            l_attn.append(attention_weight)
        return x, l_attn


class Bert(BaseModel):
    def __init__(
        self,
        img_size=256,
        hidden_dim=768,
        codebook_size=1024,
        codebook_splits=1,
        depth=24,
        heads=8,
        mlp_dim=3072,
        dropout=0.1,
        nclass=1000,
        input_stride: int = 16,
        use_prenorm: bool = False
    ):
        """ Initialize the Bert model.

            Args:
                img_size -> int: The image size. This model expects inputs of size img_size x img_size. Defaults to 256.
                hidden_dim -> int: The hidden dimension. Defaults to 768.
                codebook_size -> int: The codebook size. Defaults to 1024.
                codebook_splits -> int: The number of codebook splits. Defaults to 1.
                depth -> int: The depth of the transformer. Defaults to 24.
                heads -> int: The number of heads in the multi-head attention. Defaults to 8.
                mlp_dim -> int: The MLP dimension. Defaults to 3072.
                dropout -> float: The dropout rate. Defaults to 0.1.
                nclass -> int: The number of classes. Defaults to 1000, which is correct for ImageNet.
                input_stride -> int: The input stride. Defaults to 16.
                use_prenorm -> bool: A Flag setting prenorm or postnorm. Defaults to False.
        """
        super().__init__()
        self.nclass = nclass
        self.drop_label = nclass
        self.seq_len = (img_size // input_stride) ** 2
        self.splits = codebook_splits
        self.bits = int(math.log2(codebook_size))
        self.effective_codebook_size = int(2 ** (self.bits // self.splits))
        self.mask_token = self.effective_codebook_size

        self.class_emb = torch.nn.Embedding(nclass+1, hidden_dim) # +1 for class drop

        self.tok_emb_list = torch.nn.ModuleList()
        for _ in range(self.splits):
            self.tok_emb_list.append(torch.nn.Embedding(self.effective_codebook_size + 1, hidden_dim))  # +1 for mask token
        self.pos_emb = torch.nn.init.trunc_normal_(torch.nn.Parameter(torch.zeros(1, (self.seq_len)+1, hidden_dim)), 0., 0.02)

        # First layer before the Transformer block
        self.first_layer = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim, eps=1e-12),
            torch.nn.Dropout(p=dropout)
        )

        self.use_prenorm = use_prenorm
        self.transformer = TransformerEncoder(
            dim=hidden_dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            use_prenorm=use_prenorm
        )
        if self.use_prenorm:
            self.norm_after_transformer = torch.nn.LayerNorm(hidden_dim, eps=1e-12)
        
        # Last layer after the Transformer block
        self.last_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim, eps=1e-12),
        )

        self.bias = torch.nn.ParameterList()
        for _ in range(self.splits):
            self.bias.append(torch.nn.Parameter(torch.zeros((self.seq_len), self.effective_codebook_size)))

        self.apply(self._init_weights)

    def _init_weights(self, module: torch.nn.Module):
        """ Initialize the weights. This function is called by self.apply(...).
            Args:
                module -> torch.nn.Module: The module to initialize.
        """
        if isinstance(module, torch.nn.Linear):
            module.weight.data = torch.nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data = torch.nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_group_splits(self) -> int:
        """ Get the number of splits.
            Returns:
                int: The number of splits.
        """
        return self.splits

    def forward(
        self,
        img_tokens: torch.Tensor,
        class_labels: torch.Tensor,
        drop_label_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """ The forward pass of our Bert model.

            Args:
                img_tokens -> torch.Tensor: The image tokens.
                class_labels -> torch.Tensor: The class labels.
                drop_label_mask -> Optional[torch.Tensor]: The mask for the drop label. Defaults to None.
                return_attn -> bool: Flag setting whether to compute attention weights or not. Defaults to False.
            
            Returns:
                Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
                    If return_attn is False, returns the output Tensor of shape (b, n, c), where b is the batch size,
                    n is the sequence length, and c is the codebook size.
                    If return_attn is True, returns a tuple of (torch.Tensor, List[torch.Tensor]), where the first 
                    element is the output Tensor described above and the second element is a list of attention
                    weights.
        """
        b = img_tokens.size(0)

        cls_token = class_labels.view(b, -1)

        cls_token[drop_label_mask] = self.drop_label  # Drop condition
        cls_embedding = self.class_emb(cls_token.view(b, -1))

        tok_embeddings = self.tok_emb_list[0](img_tokens[..., 0])
        for i in range(1, self.splits):
            tok_embeddings += self.tok_emb_list[i](img_tokens[..., i])

        tok_embeddings = torch.cat([tok_embeddings, cls_embedding], dim=1)

        # Position embedding
        pos_embeddings = self.pos_emb
        x = tok_embeddings + pos_embeddings

        # transformer forward pass
        x = self.first_layer(x)
        x, attn = self.transformer(x, return_attn=return_attn)
        if self.use_prenorm:
            x = self.norm_after_transformer(x)
        x = self.last_layer(x)

        logits = []
        for i in range(self.splits):
            logit = torch.matmul(x, self.tok_emb_list[i].weight.T[:, :self.effective_codebook_size])
            logits.append(logit[:, :self.seq_len, :] + self.bias[i])

        logits = torch.stack(logits, dim=2)

        if return_attn:  # return list of attention
            return logits, attn

        return logits



class LFQBert(BaseModel):
    def __init__(
        self,
        img_size=256,
        hidden_dim=768,
        codebook_size=1024,
        codebook_splits=1,
        depth=24,
        heads=8,
        mlp_dim=3072,
        dropout=0.1,
        nclass=1000,
        input_stride: int = 16,
        use_prenorm: bool = False
    ):
        """ Initialize the Transformer model.

            Args:
                img_size -> int: The image size. This model expects inputs of size img_size x img_size. Defaults to 256.
                hidden_dim -> int: The hidden dimension. Defaults to 768.
                codebook_size -> int: The codebook size. Defaults to 1024.
                codebook_splits -> int: The number of codebook splits. Defaults to 1.
                depth -> int: The depth of the transformer. Defaults to 24.
                heads -> int: The number of heads in the multi-head attention. Defaults to 8.
                mlp_dim -> int: The MLP dimension. Defaults to 3072.
                dropout -> float: The dropout rate. Defaults to 0.1.
                nclass -> int: The number of classes. Defaults to 1000, which is correct for ImageNet.
                input_stride -> int: The input stride. Defaults to 16.
                use_prenorm -> bool: A Flag setting prenorm or postnorm. Defaults to False.
        """
        super().__init__()
        self.nclass = nclass
        self.drop_label = nclass
        self.seq_len = (img_size // input_stride) ** 2
        self.splits = codebook_splits
        self.bits = int(math.log2(codebook_size))
        effective_bits = self.bits // self.splits
        self.effective_codebook_size = int(2 ** effective_bits)
        self.mask_token = self.effective_codebook_size
        bits_to_indices = torch.pow(2.0, torch.arange(0, effective_bits))
        self.register_buffer('bits_to_indices', bits_to_indices.int())

        self.class_emb = torch.nn.Embedding(nclass+1, hidden_dim) # +1 for class drop

        self.input_proj = torch.nn.Linear(self.bits, hidden_dim)

        self.pos_emb = torch.nn.init.trunc_normal_(torch.nn.Parameter(torch.zeros(1, self.seq_len+1, hidden_dim)), 0., 0.02)

        # First layer before the Transformer block
        self.first_layer = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim, eps=1e-12),
            torch.nn.Dropout(p=dropout)
        )

        self.use_prenorm = use_prenorm
        self.transformer = TransformerEncoder(
            dim=hidden_dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            use_prenorm=use_prenorm
        )
        if self.use_prenorm:
            self.norm_after_transformer = torch.nn.LayerNorm(hidden_dim, eps=1e-12)
        
        # Last layer after the Transformer block
        self.last_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim, eps=1e-12),
        )

        self.prediction_layer = torch.nn.Linear(hidden_dim, self.splits * self.effective_codebook_size)

        self.apply(self._init_weights)

    def _init_weights(self, module: torch.nn.Module):
        """ Initialize the weights.

            Args:
                module -> torch.nn.Module: The module to initialize.
        """
        if isinstance(module, torch.nn.Linear):
            module.weight.data = torch.nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data = torch.nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_group_splits(self) -> int:
        return self.splits

    def preprocess_tokens(self, img_tokens: torch.Tensor) -> torch.Tensor:
        """ Preprocess the tokens by converting from indices to {-1,1} bits and setting masked area to 0.

            Args:
                img_tokens -> torch.Tensor: The image tokens.
            
            Returns:
                torch.Tensor: The preprocessed image tokens.
        """
        mask = img_tokens == self.mask_token
        token_as_bits = ((img_tokens[..., None].int() & self.bits_to_indices) != 0).float()
        token_as_bits = token_as_bits * 2.0 - 1.0
        token_as_bits[mask, :] = torch.full_like(token_as_bits[mask, :], 0.0)
        token_as_bits = rearrange(token_as_bits, "b n m c -> b n (m c)")
        return token_as_bits

    def forward(
        self,
        img_tokens: torch.Tensor,
        class_labels: torch.Tensor,
        drop_label_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """ Forward pass.

            Args:
                img_tokens -> torch.Tensor: The image tokens.
                class_labels -> torch.Tensor: The class labels.
                drop_label_mask -> Optional[torch.Tensor]: The mask for the drop label. Defaults to None.
                return_attn -> bool: Flag setting whether to compute attention weights or not. Defaults to False.
            
            Returns:
                Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
                    If return_attn is False, returns the output Tensor of shape (b, n, m, c), where n is the sequence length,
                    m is the number of splits, and c is the effective codebook size.
                    If return_attn is True, returns a tuple of (torch.Tensor, List[torch.Tensor]), where the first item is 
                    the output tensor described above and the second item is a list of attention weights.
        """
        b = img_tokens.size(0)

        img_tokens = self.preprocess_tokens(img_tokens)

        cls_token = class_labels.view(b, -1)

        cls_token[drop_label_mask] = self.drop_label  # Drop condition
        cls_embedding = self.class_emb(cls_token.view(b, -1))

        projected_bit_tokens = self.input_proj(img_tokens)

        projected_bit_tokens = torch.cat([projected_bit_tokens, cls_embedding], dim=1)

        # Position embedding
        pos_embeddings = self.pos_emb
        x = projected_bit_tokens + pos_embeddings

        # transformer forward pass
        x = self.first_layer(x)
        x, attn = self.transformer(x, return_attn=return_attn)
        if self.use_prenorm:
            x = self.norm_after_transformer(x)
        x = self.last_layer(x)

        logits = rearrange(self.prediction_layer(x), "b n (m c) -> b n m c", c=self.effective_codebook_size, m=self.splits)
        logits = logits[:, :self.seq_len, ...]

        if return_attn:  # return list of attention
            return logits, attn

        return logits



if __name__ == "__main__":
    bits = 18
    codebook_splits = 2
    effective_bits= bits // codebook_splits
    model = LFQBert(codebook_size=2**bits, codebook_splits=codebook_splits)
    print(model)

    batchsize = 8
    img_tokens = torch.randint(0, 2**effective_bits, (batchsize, 256, codebook_splits))
    img_tokens[:, :, 0] = torch.full_like(img_tokens, 2**effective_bits)[:, :, 0]
    y = torch.randint(0, 1000, (batchsize,))
    print(img_tokens)

    logits = model(img_tokens, y)
    print("input shape: ", img_tokens.shape)
    print("logits shape: ", logits.shape)
