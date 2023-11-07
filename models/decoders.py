from torch import nn
import torch
import torch.nn.functional as F
from models.addit_modules import SoftPositionEmbed


class BaselineDecoder(nn.Module):
    """
    Decoder architecture from https://arxiv.org/abs/1804.03599
    Code adapted from: https://github.com/1Konny/Beta-VAE/blob/master/model.py
    """

    def __init__(self, num_slots, slot_dim):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.linear1 = nn.Linear(self.slot_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 32 * 4 * 4)
        self.conv1 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.conv2 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.conv3 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.conv4 = nn.ConvTranspose2d(32, 4, 4, 2, 1)

    def forward(self, x):
        # if we are computing the jacobian we need to add a batch dim
        if len(x.shape) != 3:
            x = x.reshape(1, self.num_slots, self.slot_dim)
        for i in range(self.num_slots):
            xs = x[:, i]
            xs = self.linear1(xs)
            xs = F.elu(xs)
            xs = self.linear2(xs)
            xs = F.elu(xs)
            xs = self.linear3(xs)
            xs = F.elu(xs)
            xs = xs.view((-1, 32, 4, 4))
            xs = self.conv1(xs)
            xs = F.elu(xs)
            xs = self.conv2(xs)
            xs = F.elu(xs)
            xs = self.conv3(xs)
            xs = F.elu(xs)
            xs = self.conv4(xs)
            if i == 0:
                x_all = xs.unsqueeze(1)
            else:
                x_all = torch.cat((x_all, xs.unsqueeze(1)), dim=1)

        x_all = x_all.permute(0, 1, 3, 4, 2).flatten(0, 1)
        recons, masks = x_all.reshape(
            1, -1, x_all.shape[1], x_all.shape[2], x_all.shape[3]
        ).split([3, 1], dim=-1)
        masks = nn.Softmax(dim=1)(masks)

        xhs = recons * masks
        xh = torch.sum(xhs, dim=1).permute(0, 3, 1, 2)
        return xh


class SpatialBroadcastDecoder(nn.Module):
    """
    Spatial broadcast decoder from Slot Attention: https://arxiv.org/abs/2006.15055
    Code adapted from: https://github.com/evelinehong/slot-attention-pytorch/blob/master/model.py
    """

    def __init__(self, slot_dim, resolution, chan_dim):
        super().__init__()
        self.chan_dim = chan_dim
        self.slot_dim = slot_dim
        self.conv1 = nn.ConvTranspose2d(
            slot_dim, self.chan_dim, 5, stride=(2, 2), padding=2, output_padding=1
        )
        self.conv2 = nn.ConvTranspose2d(
            self.chan_dim, self.chan_dim, 5, stride=(2, 2), padding=2, output_padding=1
        )
        self.conv3 = nn.ConvTranspose2d(
            self.chan_dim, self.chan_dim, 5, stride=(2, 2), padding=2, output_padding=1
        )
        self.conv4 = nn.ConvTranspose2d(
            self.chan_dim, self.chan_dim, 5, stride=(2, 2), padding=2, output_padding=1
        )
        self.conv5 = nn.ConvTranspose2d(
            self.chan_dim, self.chan_dim, 5, stride=(1, 1), padding=2
        )
        self.conv6 = nn.ConvTranspose2d(self.chan_dim, 4, 3, stride=(1, 1), padding=1)
        self.decoder_initial_size = (8, 8)
        self.decoder_pos = SoftPositionEmbed(slot_dim, self.decoder_initial_size)
        self.resolution = resolution

    def forward(self, x):
        # if we are computing the jacobian we need to add a batch dim
        if len(x.shape) != 3:
            num_slots = int(x.shape[0] / self.slot_dim)
            x = x.reshape(1, num_slots, self.slot_dim)
        bs = x.shape[0]
        x = x.reshape((-1, x.shape[-1])).unsqueeze(1).unsqueeze(2)
        x = x.repeat((1, 8, 8, 1))
        x = self.decoder_pos(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = x[:, :, : self.resolution[0], : self.resolution[1]]
        x = x.permute(0, 2, 3, 1)
        recons, masks = x.reshape(bs, -1, x.shape[1], x.shape[2], x.shape[3]).split(
            [3, 1], dim=-1
        )
        masks = nn.Softmax(dim=1)(masks)
        xhs = recons * masks
        recon = torch.sum(xhs, dim=1).permute(0, 3, 1, 2)
        return recon
