import torch.nn as nn


class MyRNN(nn.Module):
    def __init__(self, vocab_size,embedding_dim, hidden_dim,padding_idx):
        super().__init__()

        # входная размерность эмбеддинг-слоя - vocab_size, выходная - hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim,padding_idx=padding_idx)

        self.lstm =  nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # self.lstm2 = nn.LSTM(embedding_dim, hidden_dim, 2, batch_first=True, dropout=0.3)

        # выходной линейный слой
        self.fc = nn.Linear(hidden_dim, vocab_size)


    def forward(self, x):
        emb = self.embedding(x) # результат эмбеддинг-слоя

        out, _ = self.lstm(emb)

        linear_out = self.fc(out)

        return linear_out
    
