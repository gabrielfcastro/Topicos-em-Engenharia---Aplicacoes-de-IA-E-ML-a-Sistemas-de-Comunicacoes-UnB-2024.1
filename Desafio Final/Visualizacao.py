import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Função para criar dados de modulação PSK
def create_psk_datasets(n_symbols, modulation):
    class PSKDataset(Dataset):
        def __init__(self, n_symbols, modulation):
            self.symbols = np.arange(0, n_symbols)
            self.modulation = modulation

        def __len__(self):
            return len(self.symbols)

        def __getitem__(self, idx):
            symbol = self.symbols[idx]
            if self.modulation == 'PSK':
                phase = 2 * np.pi * symbol / n_symbols
                return torch.Tensor([np.cos(phase), np.sin(phase)])
            else:
                raise ValueError(f"Modulação {self.modulation} não suportada.")

    dataset = PSKDataset(n_symbols, modulation)

    # Dividir o dataset em conjunto de treino e teste
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Carregar os datasets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Carregar todos os dados (útil para o espaço vetorial codificado)
    data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    # Codificador one-hot (apenas para exemplo, pode ser adaptado conforme necessário)
    class OneHotEncoder:
        def encode(self, symbol):
            return torch.Tensor([symbol])

    one_hot_encoder = OneHotEncoder()

    return train_loader, test_loader, data_loader, one_hot_encoder

# Função para criar dados de modulação QAM
def create_qam_datasets(n_symbols, modulation):
    class QAMDataset(Dataset):
        def __init__(self, n_symbols, modulation):
            self.symbols = np.arange(0, n_symbols)
            self.modulation = modulation

        def __len__(self):
            return len(self.symbols)

        def __getitem__(self, idx):
            symbol = self.symbols[idx]
            if self.modulation == 'QAM':
                i = 2 * (symbol // 2) - 1
                q = 2 * (symbol % 2) - 1
                return torch.Tensor([i, q])
            else:
                raise ValueError(f"Modulação {self.modulation} não suportada.")

    dataset = QAMDataset(n_symbols, modulation)

    # Dividir o dataset em conjunto de treino e teste
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Carregar os datasets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Carregar todos os dados (útil para o espaço vetorial codificado)
    data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    # Codificador one-hot (apenas para exemplo, pode ser adaptado conforme necessário)
    class OneHotEncoder:
        def encode(self, symbol):
            return torch.Tensor([symbol])

    one_hot_encoder = OneHotEncoder()

    return train_loader, test_loader, data_loader, one_hot_encoder
