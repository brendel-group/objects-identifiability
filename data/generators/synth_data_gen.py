import torch
import numpy as np
import utils
from scipy.stats import wishart
from torch.distributions.multivariate_normal import MultivariateNormal

def gen_synth_data(args):
    """
    Generates synthetic data from a latent variable model with a compositional generator

    Args:
        args: Command line arguments from train_model.py
        which specify: ground-truth/inferred slot dimension, number of slots, number of samples,
        and whether to sample dependent latents.

    Returns:
        numpy array of observations 'X' and corresponding latents 'Z'
    """
    if bool(args.dependent):
        # sample random covariance matrix
        sigma = wishart.rvs(args.gt_slot_dim * args.num_slots, np.eye(args.gt_slot_dim * args.num_slots), size=1)

        # sample latents from gaussian with zero mean and sampled covariance
        dist = MultivariateNormal(torch.FloatTensor([0.] * args.gt_slot_dim * args.num_slots), torch.FloatTensor(sigma))
        Z = dist.sample([args.nobs]).float()

    else:
        # sample latents from unit gaussian
        Z = torch.from_numpy(np.random.normal(0., 1., size=(args.nobs, args.num_slots*args.gt_slot_dim))).float()

    Z = Z.reshape(args.nobs, args.num_slots, args.gt_slot_dim)

    # construct ground-truth slot-wise generator
    slot_mixing = utils.construct_invertible_mlp(os_dim=args.slot_x_dim, zs_dim=args.gt_slot_dim, n_layers=2)

    # render first slot
    X = slot_mixing(Z[:, 0, :])
    for i in range(1, args.num_slots):
        # render remaining slots and concatenate outputs
        X = torch.cat((X, slot_mixing(Z[:, i, :])), 1)

    return X.float().numpy(), Z.float().numpy()