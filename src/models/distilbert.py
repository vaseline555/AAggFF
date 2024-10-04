import torch

from transformers import DistilBertModel, DistilBertConfig



class DistilBert(torch.nn.Module):
    def __init__(self, num_classes, num_embeddings, embedding_size, hidden_size, num_layers, dropout, use_pt_model, is_seq2seq):
        super(DistilBert, self).__init__()
        self.is_seq2seq = is_seq2seq

        # define encoder        
        if use_pt_model: # fine-tuning
            self.features = DistilBertModel.from_pretrained('distilbert-base-uncased')
            self.num_classes = num_classes
            self.num_embeddings = self.features.config.vocab_size
            self.embedding_size = self.features.config.dim
            self.hidden_size = self.features.config.hidden_size
            self.dropout = self.features.config.dropout

            if num_layers > 1:
                classifier = [
                    torch.nn.Linear(self.embedding_size, hidden_size), 
                    torch.nn.ReLU(), 
                    torch.nn.Dropout(dropout)
                ]
                for _ in range(num_layers - 1):
                    classifier.append(torch.nn.Linear(hidden_size, hidden_size))
                    classifier.append(torch.nn.ReLU())
                else:
                    classifier.append(torch.nn.Linear(hidden_size, num_classes))
                self.classifier = torch.nn.Sequential(*classifier)
            else:
                self.classifier = torch.nn.Sequential(
                    torch.nn.Linear(self.embedding_size, self.embedding_size, bias=True),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(self.embedding_size, self.num_classes, bias=True)
                )
        else: # from scratch
            config = DistilBertConfig(
                vocab_size=num_embeddings,
                dim=embedding_size,
            )
            self.features = DistilBertModel(config)
            if num_layers > 1:
                classifier = [
                    torch.nn.Linear(embedding_size, hidden_size), 
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout)
                ]
                for _ in range(num_layers - 1):
                    classifier.append(torch.nn.Linear(hidden_size, hidden_size))
                    classifier.append(torch.nn.ReLU(True))
                else:
                    classifier.append(torch.nn.Linear(hidden_size, num_classes))
                self.classifier = torch.nn.Sequential(*classifier)
            else:
                self.classifier = torch.nn.Sequential(
                    torch.nn.Linear(embedding_size, embedding_size, bias=True),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(embedding_size, num_classes, bias=True)
                )

    def forward(self, x):
        x = self.features(x[:, 0], attention_mask=x[:, 1])[0] if len(x.shape) > 2 else self.features(x)[0]
        x = self.classifier(x['last_hidden_state'] if self.is_seq2seq else x[:, 0, :]) # use [CLS] token for classification
        return x
    