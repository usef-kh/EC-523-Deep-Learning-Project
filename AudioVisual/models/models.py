import torch.nn as nn
import torch.nn.functional as F


# from sklearn.svm import SVC
#
# class SVM:
#     def __init__(self):
#         self.svm = SVC(kernel='rbf', C=1)
#
#     def fit(self, X, y):
#         self.svm.fit(X, y)
#
#
# class ELM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes, device=None):
#         super(ELM, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = num_classes # 2 for first ELM, 5 for second
#         self.device = device
#
#         # just two layers; declare parameters: (maybe use xavier_uniform inits?)
#         self.alpha = nn.init.uniform_(torch.empty(self.input_size, self.hidden_size, device=self.device), a=-1., b=1.)
#         self.beta = nn.init.uniform_(torch.empty(self.hidden_size, self.output_size, device=self.device), a=-1., b=1.)
#
#         self.bias = torch.zeros(self.hidden_size, device=self.device)
#
#         self.activation = torch.nn.functional.gelu() # other activations? # they said 'gaussian kernel'
#
#     def forward(self, x):
#         h = self.activation(torch.add(x.mm(self.alpha), self.bias)) # forward used for training
#         return h.mm(self.beta)
#
#     def forwardToHidden(self, x):  # the output of this is what we feed to the next ELM AFTER THIS ONE IS TRAINED
#         return self.activation(torch.add(x.mm(self.alpha), self.bias))
#
#
class CNN_2D(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=2)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(512 * 2 * 2, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 7)

        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        # print(x.shape)
        x = F.elu(self.bn1(self.conv1(x)))
        # print(x.shape)
        x = self.max_pool(x)
        # print(x.shape)
        x = F.elu(self.bn2(self.conv2(x)))
        x = self.avg_pool(x)
        # print(x.shape)
        x = F.elu(self.bn3(self.conv3(x)))
        x = self.avg_pool(x)
        # print(x.shape)
        x = F.elu(self.bn4(self.conv4(x)))
        # print(x.shape)
        x = x.view(-1, 512 * 2 * 2)
        # print(x.shape)
        x = self.drop(x)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))

        x = F.elu(self.fc3(x))

        return x


class CNN_3D(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1a = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1)

        self.conv2a = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)

        self.conv3a = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4a = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv5a = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5b = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)

        self.bn1a = nn.BatchNorm3d(64)
        self.bn2a = nn.BatchNorm3d(128)
        self.bn3a = nn.BatchNorm3d(256)
        self.bn3b = nn.BatchNorm3d(256)
        self.bn4a = nn.BatchNorm3d(256)
        self.bn4b = nn.BatchNorm3d(256)
        self.bn5a = nn.BatchNorm3d(256)
        self.bn5b = nn.BatchNorm3d(256)

        self.max_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=2)
        self.avg_pool = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=2)

        self.fc1 = nn.Linear(256 * 1 * 8 * 8, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 7)

        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        # print(x.shape)
        x = x.unsqueeze(1)
        # print(x.shape)
        x = F.elu(self.bn1a(self.conv1a(x)))
        x = self.max_pool(x)
        # print(x.shape)

        x = F.elu(self.bn2a(self.conv2a(x)))
        x = self.avg_pool(x)
        # print(x.shape)
        x = F.elu(self.bn3a(self.conv3a(x)))
        x = F.elu(self.bn3b(self.conv3b(x)))
        x = self.avg_pool(x)
        # print(x.shape)
        x = F.elu(self.bn4a(self.conv4a(x)))
        x = F.elu(self.bn4b(self.conv4b(x)))
        x = self.avg_pool(x)

        x = F.elu(self.bn5a(self.conv5a(x)))
        x = F.elu(self.bn5b(self.conv5b(x)))
        x = self.avg_pool(x)
        # print(x.shape)
        x = x.view(-1, 256 * 1 * 8 * 8)
        # print(x.shape)
        x = self.drop(x)
        # print(x.shape)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        # print(x.shape)
        x = F.elu(self.fc3(x))
        # print(x.shape)
        return x
