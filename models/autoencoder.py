import torch

class AutoEncoder(torch.nn.Module):
    def __init__(self, num_slots, slot_dim, encoder, decoder):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        """
        Encodes data batch to get inferred latents and decodes to get reconstructed observation

        Args:
            x: batch of data

        Returns:
            inferred latents and reconstructed observations
        """
        # encode
        zh = self.encoder(x)

        # decode
        xh = self.decoder(zh)

        # returns inferred latents and reconstructed observation
        return zh.reshape(zh.shape[0], self.num_slots, self.slot_dim), xh