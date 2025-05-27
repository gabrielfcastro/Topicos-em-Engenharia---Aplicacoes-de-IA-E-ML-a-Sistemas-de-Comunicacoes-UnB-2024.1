# Importando bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import math
from sklearn.manifold import TSNE
import torch
from pylab import *

# Função Q(x)
def func_q(x):
    return (1/2) * erfc((1 / math.sqrt(2)) * x)

# Função para gerar BER para modulações tradicionais
def generate_BER(M, EbNodB_range):
    m_ask = [None] * len(EbNodB_range)
    m_psk = [None] * len(EbNodB_range)
    m_qam = [None] * len(EbNodB_range)
    
    for n in range(len(EbNodB_range)):
        EbNo = 10.0 ** (EbNodB_range[n] / 10.0)
        m_ask[n] = 2 * (1 - (1 / M)) * func_q((math.sqrt((6 * math.log2(M)) / (M**2 - 1) * EbNo)))
        m_psk[n] = 2 * func_q(math.sin(math.pi / M) * math.sqrt(2 * math.log2(M) * EbNo))
        m_qam[n] = 4 * (1 - (1 / math.sqrt(M))) * func_q(math.sqrt((3 * math.log2(M) / (M - 1)) * EbNo))
    
    return m_ask, m_psk, m_qam

# Função para adicionar ruído gaussiano aos símbolos
def add_noise(scatter_plot, M, n_channel, n_net=2, EbNo=7, N_noise=1):
    x_emb = scatter_plot
    EbNo_nuvem = 10.0 ** (EbNo / 10.0)
    noise_std = np.sqrt(1 / (2 * (M / n_channel) * EbNo_nuvem)).item()
    x_parse = np.array([N_noise, n_net])
    
    for j in range(M):
        for i in range(N_noise):
            noise = torch.normal(0.0, noise_std, [1, n_net]).numpy()
            x_parse = x_emb[j, :] + noise
            if j == 0 and i == 0:
                X_embedded_noise = [x_parse]
            else:
                X_embedded_noise = np.concatenate((X_embedded_noise, [x_parse]), axis=0)
    
    return X_embedded_noise.reshape(X_embedded_noise.shape[0], 2)

# Função para reduzir as dimensões da matriz usando t-SNE
def reduce_matrix(x):
    X_embedded = TSNE(learning_rate=700, n_components=2, n_iter=35000, random_state=0, perplexity=60).fit_transform(x)
    X_embedded = (X_embedded - X_embedded.mean()) / (X_embedded.max() - X_embedded.min())
    return X_embedded

# Função para plotar a constelação com ou sem ruído
def plot_constellation(X_embedded_noise, X_embedded, flag_noise=False):
    if flag_noise:
        plt.plot(X_embedded_noise[:, 0], X_embedded_noise[:, 1], 'bo')
        plt.title('Constellation with Cloud of Noise')
    else:
        plt.plot(X_embedded[:, 0], X_embedded[:, 1], 'ro')
        plt.title('Constellation Diagram')
    
    plt.grid()
    plt.xlabel('I Axis')
    plt.ylabel('Q Axis')
    plt.show()

# Função para separar parte real e imaginária de um símbolo no espaço vetorial
def split_symbol(symbol):
    return symbol[:, ::2], symbol[:, 1::2]

# Função para plotar gráficos das constelações e indicadores
def plot_graphs(x, y, number_of_subplots=1, n_symb=0, sizefig=20, xdim=0, x_axis='Epoch', y_axis='Average Symbol Distance'):
    subplots_adjust(hspace=0.000)
    
    if number_of_subplots % 2 == 1:
        n_rows = int(number_of_subplots / 2) + number_of_subplots % 2
    else:
        n_rows = number_of_subplots / 2
    n_rows = int(n_rows)
    
    f, ax1 = plt.subplots(n_rows, 2, figsize=(sizefig, sizefig))
    
    for i in range(number_of_subplots):
        v = i + 1
        ax1 = subplot(n_rows, 2, v)
        
        if xdim == 1:
            ax1.plot(x, y[:, i], 'b')
            ax1.set_xlabel(x_axis)
            ax1.set_ylabel(y_axis)
        elif xdim == 0:
            ax1.plot(x[:, i], y[:, i], 'bo')
            ax1.plot(x[n_symb, i], y[n_symb, i], 'ro')
            ax1.set_xlabel('Real Axis')
            ax1.set_ylabel('Imaginary Axis')
        
        ax1.grid()
        channel = str(i)
        title = 'Channel Use: ' + channel
        ax1.set_title(title)
    
    f.tight_layout()
    plt.show()