import torch

from transformers import BertModel, BertConfig



class Bert(torch.nn.Module):
    def __init__(self, num_classes, num_embeddings, embedding_size, num_layers, hidden_size, dropout, use_pt_model, is_seq2seq):
        super(Bert, self).__init__()
        self.is_seq2seq = is_seq2seq
        
        # define encoder        
        if use_pt_model: # fine-tuning
            self.features = BertModel.from_pretrained('bert-base-uncased')
            self.hidden_size = self.features.config.hidden_size 
            self.classifier = torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                torch.nn.Linear(self.features.config.hidden_size, num_classes, bias=True)
            )

        else: # from scratch
            self.num_classes = num_classes
            self.num_embeddings = num_embeddings
            self.embedding_size = embedding_size
            self.num_hiddens = hidden_size
            self.num_layers = num_layers
            self.dropout = dropout

            config = BertConfig(
                vocab_size=self.num_embeddings,
                hidden_size=self.embedding_size,
                intermediate_size=self.num_hiddens,
                num_hidden_layers=self.num_layers,
                hidden_dropout_prob=self.dropout
            )
            self.features = BertModel(config)
            self.classifier = torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                torch.nn.Linear(self.pretrained.config.hidden_size, self.num_classes, bias=True)
            )

    def forward(self, x):
        x = self.features(x[:, 0], attention_mask=x[:, 1]) if len(x.shape) > 2 else self.features(x)
        x = self.classifier(x['last_hidden_state'] if self.is_seq2seq else x['pooler_output'])
        return x
