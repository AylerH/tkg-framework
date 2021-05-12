from sklearn.manifold import TSNE
from sklearn.datasets import load_iris, load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import torch
import umap
import umap.utils as utils
import umap.aligned_umap
import sklearn.decomposition

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from tkge.common.config import Config
from tkge.data.dataset import DatasetProcessor

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# datasets
config = '/Users/GengyuanMax/workspace/tkg-framework/config-hpo-hytetranse.yaml'
config = Config.create_from_yaml(config)
dataset = DatasetProcessor.create(config=config)

china = [v for k, v in dataset.ent2id.items() if 'China' in k]
america = [v for k, v in dataset.ent2id.items() if 'United States' in k]

all = (china + america)
all.sort()
print(all)

cl =[]
for i in all:
    if i in china:
        cl.append('red')
    else:
        cl.append('blue')

path = '/Users/GengyuanMax/Downloads/translation_rtranse_best.ckpt'
ckpt = torch.load(path, map_location=torch.device('cpu'))
head_ent_emb = ckpt['state_dict']['_entity_embeddings._tail.real.weight'].numpy()
temp_emb = ckpt['state_dict']['_temporal_embeddings._temporal.real.weight'].numpy()

# head_ent_emb0 = head_ent_emb # + temp_emb[120:121, :]
# X_pca = PCA(n_components=2).fit_transform(head_ent_emb0)
#
# plt.figure(figsize=(16, 40))


head_ent_emb_part = head_ent_emb[all, :]

relation_dict = {i: i for i in range(100)}
relation_dicts = [relation_dict.copy() for i in range(12 - 1)]


slices = []


for i in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
    head_ent_emb_part_at_i = head_ent_emb_part + temp_emb[i:i + 1, :]

    slices.append(head_ent_emb_part_at_i)

aligned_mapper = umap.AlignedUMAP().fit(slices, relations=relation_dicts)



def axis_bounds(embedding):
    left, right = embedding.T[0].min(), embedding.T[0].max()
    bottom, top = embedding.T[1].min(), embedding.T[1].max()
    adj_h, adj_v = (right - left) * 0.1, (top - bottom) * 0.1

    return [left - adj_h, right + adj_h, bottom - adj_v, top + adj_v]

fig, axs = plt.subplots(6,2, figsize=(10, 20))
ax_bound = axis_bounds(np.vstack(aligned_mapper.embeddings_))
for i, ax in enumerate(axs.flatten()):
    current_target = 'red'
    ax.scatter(*aligned_mapper.embeddings_[i].T, s=2, c=cl, cmap="Spectral")
    ax.axis(ax_bound)
    ax.set(xticks=[], yticks=[])
plt.tight_layout()

plt.show()

# #[2, 5, 10, 30, 50, 100]
# for i, n in enumerate([5, 15, 30, 50, 60, 80]):
#     X_tsne = TSNE(n_components=2,perplexity=n,random_state=13,n_iter=1000).fit_transform(head_ent_emb0)
#
#
#     #
#     plt.subplot(6,2,2*i+1)
#     plt.scatter(X_tsne[:, 0], X_tsne[:, 1],label="t-SNE")
#     plt.legend()
#     plt.subplot(6,2,2*i+2)
#     plt.scatter(X_pca[:, 0], X_pca[:, 1],label="PCA")
#     plt.legend()
#
# plt.show()
