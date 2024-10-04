import torch

from transformers import SqueezeBertModel, SqueezeBertConfig



class SqueezeBert(torch.nn.Module):
    def __init__(self, num_classes, num_embeddings, embedding_size, hidden_size, num_layers, dropout, use_pt_model, is_seq2seq):
        super(SqueezeBert, self).__init__()
        self.is_seq2seq = is_seq2seq

        # define encoder        
        if use_pt_model: # fine-tuning
            self.features = SqueezeBertModel.from_pretrained('squeezebert/squeezebert-uncased')
            self.num_classes = num_classes
            self.num_embeddings = self.features.config.vocab_size
            self.embedding_size = self.features.config.embedding_size
            self.hidden_size = self.features.config.hidden_size
            self.dropout = self.features.config.hidden_dropout_prob 
            
            if num_layers > 1:
                classifier = [
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(self.embedding_size, hidden_size), 
                    torch.nn.ReLU(True), 
                ]
                for _ in range(num_layers - 1):
                    classifier.append(torch.nn.Linear(hidden_size, hidden_size))
                    classifier.append(torch.nn.ReLU(True))
                else:
                    classifier.append(torch.nn.Linear(hidden_size, num_classes))
                self.classifier = torch.nn.Sequential(*classifier)
            else:
                self.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(self.embedding_size, self.num_classes, bias=True)
                )
        else: # from scratch
            assert embedding_size == hidden_size, 'If you want embedding_size != intermediate hidden_size, please insert a Conv1d layer to adjust the number of channels before the first SqueezeBertModule.'
            config = SqueezeBertConfig(
                vocab_size=num_embeddings,
                embedding_size=embedding_size,
                hidden_size=hidden_size,
                hidden_dropout_prob=dropout
            )
            self.features = SqueezeBertModel(config)
            if num_layers > 1:
                classifier = [
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(embedding_size, hidden_size), 
                    torch.nn.ReLU(True)
                ]
                for _ in range(num_layers - 1):
                    classifier.append(torch.nn.Linear(hidden_size, hidden_size))
                    classifier.append(torch.nn.ReLU(True))
                else:
                    classifier.append(torch.nn.Linear(hidden_size, num_classes))
                self.classifier = torch.nn.Sequential(*classifier)
            else:
                self.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(self.embedding_size, self.num_classes, bias=True)
                )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x['last_hidden_state'] if self.is_seq2seq else x['pooler_output'])
        return x
