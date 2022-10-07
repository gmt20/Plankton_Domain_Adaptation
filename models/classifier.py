import torch.nn.functional as F
from torch import nn
from cba.har.utils.types import EncoderType


class Classifier(nn.Module):  ###classifier head for cross entropy loss
    def __init__(
        self, embedding_dimension, num_of_classes,ln1=256, p=0.2
    ):
        super(Classifier, self).__init__()
        # Defining the two layer MLP
        ln2 = ln1 // 2
        self.embedding_dimension = embedding_dimension
        self.softmax = nn.Sequential(
            nn.Linear(embedding_dimension, ln1),
            nn.BatchNorm1d(ln1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(ln1, ln2),
            nn.BatchNorm1d(ln2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(ln2, num_of_classes),
        )

        def _weights_init(m):
            if isinstance(m, nn.Conv1d or nn.Linear or nn.LSTM):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.apply(_weights_init)

    def forward(self, encoding):

        out = self.softmax(encoding)

        return out
