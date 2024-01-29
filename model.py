'''
https://keras.io/examples/vision/attention_mil_classification/
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleAttentionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.M = 500 ## embedding space dims
        self.L = 128 ## hyperparams for attention
        self.ATTENTION_BRANCHES = 1

        self.feature_extractor1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        ) ## architecture from LeNet

        self.feature_extractor2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.M),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor1(x)
        H = torch.flatten(H, start_dim=1)
        H = self.feature_extractor2(H) ## (K, M)

        A = self.attention(H) # (K, ATTENTION_BRANCHES)
        A = torch.transpose(A, 1, 0) # (ATTENTION_BRANCHES, K)
        A = A.softmax(dim=1) # softmax over K 

        Z = torch.mm(A, H) # (ATTENTION_BRANCHES, M)

        logit = self.classifier(Z)

        return logit, A



if __name__ == '__main__':
    x = torch.randn(size=(1, 3, 1, 28, 28))
    mdoel = SimpleAttentionCNN()
    print(mdoel(x).shape)