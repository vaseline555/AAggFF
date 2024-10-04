import torch

from src.models.model_utils import Lambda



class StackedGRU(torch.nn.Module):
    def __init__(self, num_classes, embedding_size, num_embeddings, hidden_size, dropout, num_layers, is_seq2seq, need_embedding):
        super(StackedGRU, self).__init__()
        self.is_seq2seq = is_seq2seq
        self.num_hiddens = hidden_size
        self.num_classes = num_classes
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.num_layers = num_layers

        self.features = torch.nn.Sequential(
            torch.nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_size) if need_embedding else torch.nn.Identity(),
            torch.nn.GRU(
                input_size=self.embedding_size,
                hidden_size=self.num_hiddens,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout,
                bias=True
            ),
            Lambda(lambda x: x[0])
        )
        self.classifier = torch.nn.Linear(self.num_hiddens, self.num_classes, bias=True)

    def forward(self, x):
        x = self.features(x.permute(0, 2, 1))
        x = self.classifier(x if self.is_seq2seq else x[:, -1, :])
        return x
