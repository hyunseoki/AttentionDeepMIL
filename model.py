'''
https://keras.io/examples/vision/attention_mil_classification/
https://github.com/AMLab-Amsterdam/AttentionDeepMI
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMILNet(nn.Module):
    def __init__(self, gate=False):
        super().__init__()
        self.M = 500 ## embedding space dims
        self.L = 128 ## hyperparams for attention
        self.ATTENTION_BRANCHES = 1
        self.gate = gate

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

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Tanh()
        )
        self.attention_W = nn.Linear(self.L, self.ATTENTION_BRANCHES)
        if gate:
            self.attention_U = nn.Sequential(
                nn.Linear(self.M, self.L),
                nn.Sigmoid()
            )

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )

    def attention(self, H):
        A_V = self.attention_V(H)

        if self.gate:
            A_U = self.attention_U(H)
            A = self.attention_W(A_V * A_U) # element wise multiplication

        else:
            A = self.attention_W(A_V)

        A = torch.transpose(A, 1, 0) # (ATTENTION_BRANCHES, K)
        A = A.softmax(dim=1) # softmax over K 

        return A


    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor1(x)
        H = torch.flatten(H, start_dim=1)
        H = self.feature_extractor2(H) ## (K, M)

        A = self.attention(H) # (K, ATTENTION_BRANCHES)
        Z = torch.mm(A, H) # (ATTENTION_BRANCHES, M)
        logit = self.classifier(Z)

        return logit, A



if __name__ == '__main__':
    x = torch.randn(size=(1, 3, 1, 28, 28))
    mdoel = AttentionMILNet(gate=True)
    print(mdoel(x)[0].shape)