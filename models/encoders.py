from torch import nn
import torch
import torch.nn.functional as F
from models.addit_modules import SlotAttention
from models.addit_modules import SoftPositionEmbed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MonolithicEncoder(nn.Module):
    """
    Encoder architecture from https://arxiv.org/abs/1804.03599
    Code adapted from: https://github.com/1Konny/Beta-VAE/blob/master/model.py
    """

    def __init__(self, num_slots, hid_dim, nc=3):
        super().__init__()
        self.num_slots = num_slots
        self.hid_dim = hid_dim
        self.conv1 = nn.Conv2d(nc, 32, 4, 2, 1)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)
        self.conv3 = nn.Conv2d(32, 32, 4, 2, 1)
        self.conv4 = nn.Conv2d(32, 32, 4, 2, 1)
        self.linear1 = nn.Linear(32 * 4 * 4, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, self.num_slots * self.hid_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.elu(x)
        x = self.conv2(x)
        x = F.elu(x)
        x = self.conv3(x)
        x = F.elu(x)
        x = self.conv4(x)
        x = F.elu(x)
        x = x.view((-1, 32 * 4 * 4))
        x = self.linear1(x)
        x = F.elu(x)
        x = self.linear2(x)
        x = F.elu(x)
        x = self.linear3(x)
        return x.reshape(x.shape[0], self.num_slots, self.hid_dim)


class SlotEncoder(nn.Module):
    """
    Convolutional encoder from Slot Attention: https://arxiv.org/abs/2006.15055
    Code adapted from: https://github.com/evelinehong/slot-attention-pytorch/blob/master/model.py
    """

    def __init__(self, resolution, num_slots, slot_dim, chan_dim):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.chan_dim = chan_dim
        self.conv1 = nn.Conv2d(3, self.chan_dim, 5, padding=2)
        self.conv2 = nn.Conv2d(self.chan_dim, self.chan_dim, 5, padding=2)
        self.conv3 = nn.Conv2d(self.chan_dim, self.chan_dim, 5, padding=2)
        self.conv4 = nn.Conv2d(self.chan_dim, slot_dim, 5, padding=2)
        self.fc1 = nn.Linear(slot_dim, slot_dim)
        self.fc2 = nn.Linear(slot_dim, slot_dim)
        self.encoder_pos = SoftPositionEmbed(self.slot_dim, resolution)

        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            dim=self.slot_dim,
            iters=3,
            eps=1e-8,
            hidden_dim=64,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = x.permute(0, 2, 3, 1)
        x = self.encoder_pos(x)
        x = torch.flatten(x, 1, 2)
        x = nn.LayerNorm(x.shape[1:]).to(device)(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return self.slot_attention(x)
