import torch
import torch.nn as nn


class ViT(nn.Module):
    def __init__(self, image_size, num_channels, patch_size, num_classes, embedding_dim, num_head, num_layers):
        super(ViT, self).__init__()

        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = patch_size ** 2 * num_channels

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embedding_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.patch_embedding = nn.Linear(self.patch_dim, embedding_dim)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(embedding_dim, nhead=num_head),
                                                 num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(-1, self.num_patches, self.patch_dim)
        x = self.patch_embedding(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.position_embedding
        x = x.permute(1, 0, 2)

        x = self.transformer(x)
        x = x.mean(dim=0)
        x = self.fc(x)

        return x