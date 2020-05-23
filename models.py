import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, embedding_size, num_embeddings, num_channels,
                 hidden_dim, num_classes, dropout_p,
                 pretrained_embeddings=None, padding_idx=0):
        """
        Args:
            embedding_size (int): size of the embedding vectors
            num_embeddings (int): number of embedding vectors
            filter_width (int): width of the convolutional kernels
            num_channels (int): number of convolutional kernels per layer
            hidden_dim (int): the size of the hidden dimension
            num_classes (int): the number of classes in classification
            dropout_p (float): a dropout parameter
            pretrained_embeddings (numpy.array): previously trained word embeddings
                default is None. If provided,
            padding_idx (int): an index representing a null position
        """
        super(SimpleCNN, self).__init__()

        if pretrained_embeddings is None:

            self.emb = nn.Embedding(embedding_dim=embedding_size,
                                    num_embeddings=num_embeddings,
                                    padding_idx=padding_idx)
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.emb = nn.Embedding(embedding_dim=embedding_size,
                                    num_embeddings=num_embeddings,
                                    padding_idx=padding_idx,
                                    _weight=pretrained_embeddings)

        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=embedding_size,
                      out_channels=num_channels, kernel_size=3),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                      kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                      kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                      kernel_size=3),
            nn.ELU()
        )

        self._dropout_p = dropout_p
        self.fc1 = nn.Linear(num_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_in, apply_softmax=False):
        """The forward pass of the classifier

        Args:
            x_in (torch.Tensor): an input data tensor.
                x_in.shape should be (batch, dataset._max_seq_length)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, num_classes)
        """

        # embed and permute so features are channels
        x_embedded = self.emb(x_in).permute(0, 2, 1)

        features = self.convnet(x_embedded)

        # average and remove the extra dimension
        remaining_size = features.size(dim=2)
        features = F.avg_pool1d(features, remaining_size).squeeze(dim=2)
        features = F.dropout(features, p=self._dropout_p)

        # mlp classifier
        intermediate_vector = F.relu(F.dropout(self.fc1(features), p=self._dropout_p))
        prediction_vector = self.fc2(intermediate_vector)

        if apply_softmax:
            prediction_vector = F.softmax(prediction_vector, dim=1)

        return prediction_vector


class VDCNN(nn.Module):
    """VDCNN class"""

    def __init__(self, num_classes: int, embedding_dim: int, k_max: int, embedding_size: int) -> None:
        """Instantiating VDCNN class

        Args:
            num_classes (int): the number of classes
            embedding_dim (int): embedding dimension of token
            k_max (int): parameter of k-max pooling following last convolution block
            embedding_size (dict): token2idx
        """
        super(VDCNN, self).__init__()

        self._extractor = nn.Sequential(nn.Embedding(embedding_size, embedding_dim, 1),
                                        Permute(),
                                        nn.Conv1d(embedding_dim, 64, 3, 1, 1),
                                        ConvBlock(64, 64),
                                        ConvBlock(64, 64),
                                        nn.MaxPool1d(2, 2),
                                        ConvBlock(64, 128),
                                        ConvBlock(128, 128),
                                        nn.MaxPool1d(2, 2),
                                        ConvBlock(128, 256),
                                        ConvBlock(256, 256),
                                        nn.MaxPool1d(2, 2),
                                        ConvBlock(256, 512),
                                        ConvBlock(512, 512),
                                        nn.AdaptiveMaxPool1d(k_max),
                                        Flatten())

        self._classifier = nn.Sequential(nn.Linear(512 * k_max, 2048),
                                         nn.ReLU(),
                                         nn.Linear(2048, 2048),
                                         nn.ReLU(),
                                         nn.Linear(2048, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self._extractor(x)
        score = self._classifier(feature)
        return score


class Flatten(nn.Module):
    """Flatten class"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class Permute(nn.Module):
    """Permute class"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1)


class ConvBlock(nn.Module):
    """ConvBlock class"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Instantiating ConvBlock class

        Args:
            in_channels (int): in_channels of ConvBlock
            out_channels (int): out_channels of ConvBlock
        """
        super(ConvBlock, self).__init__()
        self._projection = (in_channels != out_channels)
        self._ops = nn.Sequential(nn.Conv1d(in_channels, out_channels, 3, 1, 1),
                                  nn.BatchNorm1d(out_channels),
                                  nn.ReLU(),
                                  nn.Conv1d(out_channels, out_channels, 3, 1, 1),
                                  nn.BatchNorm1d(out_channels),
                                  nn.ReLU())

        if self._projection:
            self._shortcut = nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, 1),
                                           nn.BatchNorm1d(out_channels))
        self._bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self._shortcut(x) if self._projection else x
        fmap = self._ops(x) + shortcut
        fmap = F.relu(self._bn(fmap))
        return fmap
