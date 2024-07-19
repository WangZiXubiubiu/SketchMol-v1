import torch
import torch.nn as nn
from functools import partial
import clip
from einops import rearrange, repeat
import numpy as np
import math
from ldm.modules.x_transformer import Encoder, TransformerWrapper, \
    AttentionLayers

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class MixedEmbedderV2(nn.Module):
    def __init__(self, embed_dim, n_classes=100, n_layers=1, device="cuda"):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.device = device
        self.n_layers = n_layers

        self.logp_embed = nn.Linear(1, embed_dim)
        self.qed_embed = nn.Linear(1, embed_dim)
        self.sa_embed = nn.Linear(1, embed_dim)
        self.weight_embed = nn.Linear(1, embed_dim)
        self.tpsa_embed = nn.Linear(1, embed_dim)
        self.hbd_embed = nn.Linear(1, embed_dim)
        self.hba_embed = nn.Linear(1, embed_dim)
        self.rotatable_embed = nn.Linear(1, embed_dim)
        self.propery_embed_list = [None, None, self.logp_embed, self.qed_embed, self.sa_embed,
                                   self.weight_embed, self.tpsa_embed, self.hbd_embed,
                                   self.hba_embed, self.rotatable_embed]

        self.attention_layer = Encoder(dim=embed_dim, depth=self.n_layers)

    def forward(self, dict_of_batch):
        # this is a combined encoder
        # discrete property and continuous property
        # finally combine them together and input into a transformer
        various_conditions = dict_of_batch["various_conditions"]
        various_conditions_discrete = dict_of_batch["various_conditions_discrete"]
        cur_batch_len = various_conditions.shape[0]
        all_embeds = []  # batch, condition, embed
        for cur_sample in range(cur_batch_len):
            cur_embeds = []
            for idx, (feature, is_discrete) in enumerate(
                    zip(various_conditions[cur_sample], various_conditions_discrete[cur_sample])):
                if is_discrete:
                    cur_embeds.append(self.embedding(feature.to(torch.long)))
                else:
                    cur_embeds.append(self.propery_embed_list[idx](feature.unsqueeze(0)))
            all_embeds.append(torch.stack(cur_embeds, dim=0))

        all_embeds = torch.stack(all_embeds, dim=0).to(self.device)
        attned_embeds = self.attention_layer(all_embeds)
        return attned_embeds

    def encode(self, dict_of_batch):
        return self(dict_of_batch)


class MixedEmbedder_single_protein(MixedEmbedderV2):
    def __init__(self, embed_dim, n_classes=100, n_layers=1, device="cuda"):
        super().__init__(embed_dim, n_classes, n_layers, device)
        print("prepare protein embedder")

        self.propery_embed_list = self.propery_embed_list + [None] * 2




