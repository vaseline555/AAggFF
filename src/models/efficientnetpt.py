import torch

from torchvision.models import efficientnet_b0


class EfficientNetPT(torch.nn.Module): 
    def __init__(self, in_channels, hidden_size, num_layers, dropout, num_classes, use_pt_model):
        super(EfficientNetPT, self).__init__()
        assert in_channels == 3, 'EfficientNet is for a three-channel input!'
        self.features = efficientnet_b0(weights='DEFAULT' if use_pt_model else None).features
        _feature_size = self.features[-1][0].out_channels * (7 * 7)
        
        if num_layers > 1:
            classifier = [
                torch.nn.AdaptiveAvgPool2d((7, 7)),
                torch.nn.Flatten(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(_feature_size, hidden_size), 
                torch.nn.ReLU(True), 
            ]
            for _ in range(num_layers - 1):
                classifier.append(torch.nn.Linear(hidden_size, hidden_size))
                classifier.append(torch.nn.ReLU())
            else:
                classifier.append(torch.nn.Linear(hidden_size, num_classes))
            self.classifier = torch.nn.Sequential(*classifier)
        else:
            self.classifier = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d((7, 7)),
                torch.nn.Flatten(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(_feature_size, num_classes, bias=True)
            )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
