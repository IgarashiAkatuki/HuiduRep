import random

import numpy as np
import torch
import umap
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
sns.set_style("whitegrid")

label_encoder = LabelEncoder()

def generate_testset(test_data, test_labels, num_units=10, labels=None):

    if labels is None or not isinstance(labels, (np.ndarray, list)):
        labels = np.unique(test_labels)
        samples = [i for i in range(len(labels))]
        samples = random.sample(samples, num_units)
        labels = labels[samples]

    mask = torch.isin(torch.tensor(test_labels), torch.tensor(labels))
    indices = torch.nonzero(mask, as_tuple=False).squeeze(dim=1)

    test_data = test_data[indices]
    test_labels = test_labels[indices]
    # test_labels = label_encoder.fit_transform(test_labels)
    return test_data, test_labels, labels


def visualize_tsne(emb_train, emb_unseen_raw, emb_unseen_denoised, save_path=None):
    """
    将三个 embedding 数据做 t-SNE 降维并可视化。
    输入维度应为 [N, D]。
    """
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42)
    all_data = np.vstack([emb_train, emb_unseen_raw, emb_unseen_denoised])
    tsne_result = tsne.fit_transform(all_data)

    n1 = emb_train.shape[0]
    n2 = emb_unseen_raw.shape[0]
    n3 = emb_unseen_denoised.shape[0]

    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_result[:n1, 0], tsne_result[:n1, 1], label='Train', alpha=0.5, s=20, c='blue')
    plt.scatter(tsne_result[n1:n1+n2, 0], tsne_result[n1:n1+n2, 1], label='Unseen Raw', alpha=0.5, s=20, c='red')
    plt.scatter(tsne_result[n1+n2:, 0], tsne_result[n1+n2:, 1], label='Unseen Denoised', alpha=0.5, s=20, c='green')

    plt.title("t-SNE of Train / Unseen / Denoised Embeddings")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=1000)
    plt.show()

def visualize_umap(emb_train, emb_unseen_raw, emb_unseen_denoised, save_path=None):
    """
    使用 UMAP 可视化三类 latent 表征。

    参数：
        emb_train: np.ndarray [N1, D]，训练集表征
        emb_unseen_raw: np.ndarray [N2, D]，zero-shot 原始输入
        emb_unseen_denoised: np.ndarray [N3, D]，zero-shot 降噪后的输入
    """
    reducer = PCA(n_components=2, random_state=42)
    # 拼接所有数据
    reducer.fit(emb_train)
    emb_train = reducer.transform(emb_train)
    emb_unseen_raw = reducer.transform(emb_unseen_raw)
    emb_unseen_denoised = reducer.transform(emb_unseen_denoised)

    # scaler = StandardScaler()
    # emb_train = scaler.fit_transform(emb_train)
    # emb_unseen_raw = scaler.transform(emb_unseen_raw)
    # emb_unseen_denoised = scaler.transform(emb_unseen_denoised)

    all_data = np.vstack([emb_train, emb_unseen_raw, emb_unseen_denoised])

    # embedding = reducer.fit_transform(all_data)
    embedding = all_data
    n1 = emb_train.shape[0]
    n2 = emb_unseen_raw.shape[0]
    n3 = emb_unseen_denoised.shape[0]

    color = ["#43978F", "#9EC4BE", "#ABD0F1", "#DCE9F4", "#E56F5E", "#F19685", "#F6C957", "#FFB77F"]

    train_center = np.mean(emb_train, axis=0)
    test_raw_center = np.mean(emb_unseen_raw, axis=0)
    test_denoised_center = np.mean(emb_unseen_denoised, axis=0)

    plt.figure(figsize=(6, 6))



    plt.scatter(embedding[:n1, 0], embedding[:n1, 1], c=color[0], alpha=0.4, s=15, label='Train Data')
    plt.scatter(embedding[n1:n1 + n2, 0], embedding[n1:n1 + n2, 1], c=color[2], alpha=0.4, s=15, label='ID Data')
    plt.scatter(embedding[n1 + n2:, 0], embedding[n1 + n2:, 1], c=color[4], alpha=0.4, s=15, label='Denoised ID Data')

    plt.scatter(train_center[0], train_center[1], c='black', marker='X', alpha=1, s=50, zorder=1000)
    plt.text(train_center[0] + 1, train_center[1], 'Train Data Center', fontsize=16)

    plt.scatter(test_raw_center[0], test_raw_center[1], c='black',  marker='X',alpha=1, s=50, zorder=1000)
    # plt.text(test_raw_center[0] + 1, test_raw_center[1], 'Unseen Data Center', fontsize=12)
    plt.text(test_raw_center[0] + 1, test_raw_center[1] - 3, 'ID Data Center', fontsize=16)

    plt.scatter(test_denoised_center[0], test_denoised_center[1], c='black', marker='X', alpha=1, s=50, zorder=1000)
    # plt.text(test_denoised_center[0] + 1, test_denoised_center[1], 'Denoised Unseen Data Center', fontsize=12)
    plt.text(test_denoised_center[0] + 1, test_denoised_center[1] - 2, 'Denoised ID Data Center', fontsize=16)
    plt.title("IBL_TEST_DATASET Embeddings (PCA)", fontsize=18)
    plt.xlabel('PC 1', fontsize=18)
    plt.ylabel('PC 2', fontsize=18)
    plt.xlim(-40, 40)
    plt.ylim(-20, 20)
    plt.legend(loc='upper right', fontsize=18)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=1000)
    plt.show()

    return embedding[:n1, :], embedding[n1:n1 + n2, :], embedding[n1 + n2:, :]

