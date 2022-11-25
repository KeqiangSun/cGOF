import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.to(x.device)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.to(x.device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, device, global_feat=True, output_dim=1024, pointfeat_dim=64,
                    feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.device = device
        # self.output_dim = output_dim if global_feat else output_dim + pointfeat_dim
        self.output_dim = output_dim
        self.pointfeat_dim = pointfeat_dim
        
        self.stn = STN3d(channel).to(self.device)
        self.conv1 = torch.nn.Conv1d(channel, pointfeat_dim, 1).to(self.device)
        self.conv2 = torch.nn.Conv1d(pointfeat_dim, 128, 1).to(self.device)
        self.conv3 = torch.nn.Conv1d(128, output_dim, 1).to(self.device)
        self.bn1 = nn.BatchNorm1d(pointfeat_dim).to(self.device)
        self.bn2 = nn.BatchNorm1d(128).to(self.device)
        self.bn3 = nn.BatchNorm1d(output_dim).to(self.device)
        
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=pointfeat_dim).to(self.device)

    def forward(self, x):
        x = x.transpose(2, 1)
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.output_dim)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, self.output_dim, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class SimplePointNetEncoder(nn.Module):
    def __init__(self, global_feat=False, output_dim=1024, pointfeat_dim=64, channel=3):
        super(SimplePointNetEncoder, self).__init__()
        self.global_feat = global_feat
        self.output_dim = output_dim if global_feat else output_dim + pointfeat_dim
        self.pointfeat_dim = pointfeat_dim

        self.conv1 = torch.nn.Conv1d(channel, pointfeat_dim, 1)
        self.conv2 = torch.nn.Conv1d(pointfeat_dim, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, output_dim, 1)
        self.bn1 = nn.BatchNorm1d(pointfeat_dim)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = x.transpose(2, 1)
        
        B, D, N = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        if self.global_feat:
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, self.output_dim)
        else:
            x = torch.cat([x, pointfeat], 1)
            x = x.transpose(2, 1)
            
        return x


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

if __name__ == "__main__":
    model = SimplePointNetEncoder(global_feat=True)
    v = torch.rand(8, 33744, 3)
    output = model(v)
    print("output shape when global_feat = True:", output.shape)
    
    model = SimplePointNetEncoder(global_feat=False)
    v = torch.rand(8, 33744, 3)
    output = model(v)
    print("output shape when global_feat = False:", output.shape)