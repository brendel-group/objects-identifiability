import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
import random
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from data.dataloader import ObjectsDataset
from models import encoders, decoders
from models.autoencoder import AutoEncoder
from torch import nn
import hydra
import omegaconf
from models.addit_modules import MLP
from data.generators.synth_data_gen import gen_synth_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    """
    Fixes random seed

    Args:
        seed: random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_direcs(args, seed):
    """
    Setups directory to save model logs

    Args:
        args: Command line arguments from train_model.py

    Returns:
        directory for model logs as a string
    """
    model_dir = "logs/"+args.data+"_"+str(args.num_slots)+"_"+str(args.inf_slot_dim)+"/enc_"+args.encoder+"_dec_"+args.decoder+"_lambda_"+str(args.lam)+"/saved_models/"+"seed_"+str(seed)+"/"

    if not os.path.exists(model_dir):
      os.makedirs(model_dir)

    return model_dir


def get_data(args):
    """
    Generates or loads pre-generated dataset and creates training and validation dataloaders

    Args:
        args: Command line arguments from train_model.py

    Returns:
        train and validation PyTorch Dataloaders
    """

    # generate or load data depending on data type
    if args.data == "synth":
        X, Z = gen_synth_data(args)
        transform = None
        val_batch = 100

    elif args.data == "spriteworld":
        data_path = "data/datasets/"+str(args.num_slots)+"_obj_sprites.npz"
        X = np.load(data_path)['arr_0']
        Z = np.load(data_path)['arr_1']
        transform = transforms.ToTensor()
        val_batch = 5

    # train and validation splits
    Z_train, Z_val = np.split(Z, [int(.9 * len(Z))])
    X_train, X_val = np.split(X, [int(.9 * len(Z))])

    # create dataloaders
    train_loader = DataLoader(ObjectsDataset(X_train, Z_train, transform=transform), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ObjectsDataset(X_val, Z_val, transform=transform), batch_size=val_batch, shuffle=True)

    return train_loader, val_loader


def get_model(args):
    """
    Creates model based on command line arguments in train_model.py

    Args:
        args: Command line arguments from train_model.py

    Returns:
        PyTorch model
    """

    # get monet model if encoder or decoder specified as monet
    if args.encoder == "monet" or args.decoder == "monet":
        config = omegaconf.OmegaConf.load("models/monet/monet.yaml")
        return hydra.utils.instantiate(config.model).to(device)

    if args.data == "spriteworld":
        resolution = (64, 64)
        chan_dim = 16

    # get encoder
    if args.data == "synth":
        encoder = MLP(input_dim = args.slot_x_dim * args.num_slots, output_dim = args.inf_slot_dim * args.num_slots).to(device)
    elif args.encoder == "monolithic":
        encoder = encoders.MonolithicEncoder(num_slots = args.num_slots, hid_dim=args.inf_slot_dim).to(device)
    elif args.encoder == "slot-attention":
        encoder = encoders.SlotEncoder(resolution = resolution, num_slots= args.num_slots, slot_dim=args.inf_slot_dim, chan_dim=chan_dim).to(device)

    # get decoder
    if args.data == "synth":
        decoder = MLP(input_dim=args.inf_slot_dim * args.num_slots, output_dim=args.slot_x_dim * args.num_slots).to(device)
    if args.decoder == "baseline":
        decoder = decoders.BaselineDecoder(num_slots = args.num_slots, slot_dim = args.inf_slot_dim).to(device)
    elif args.decoder == "spatial-broadcast":
        decoder = decoders.SpatialBroadcastDecoder(slot_dim=args.inf_slot_dim, resolution=resolution, chan_dim=chan_dim).to(device)

    # get autoencoder
    autoencoder = AutoEncoder(num_slots=args.num_slots, slot_dim = args.inf_slot_dim, encoder = encoder, decoder = decoder).to(device)
    return autoencoder


def hungarian_algorithm(cost_matrix):

    """
    Batch-applies the hungarian algorithm to find a matching that minimizes the overall cost.
    Code adapted from: https://github.com/addtt/object-centric-library/blob/main/utils/slot_matching.py

    Returns the matching indices as a LongTensor with shape (batch size, 2, min(num objects, num slots)).
    The first column is the row indices (the indices of the true objects) while the second
    column is the column indices (the indices of the slots). The row indices are always
    in ascending order, while the column indices are not necessarily.

    The outputs are on the same device as `cost_matrix` but gradients are detached.

    A small example:
                | 4, 1, 3 |
                | 2, 0, 5 |
                | 3, 2, 2 |
                | 4, 0, 6 |
    would result in selecting elements (1,0), (2,2) and (3,1). Therefore, the row
    indices will be [1,2,3] and the column indices will be [0,2,1].

    Args:
        cost_matrix: Tensor of shape (batch size, num objects, num slots).

    Returns:
        A tuple containing:
            - a Tensor with shape (batch size, min(num objects, num slots)) with the
              costs of the matches.
            - a LongTensor with shape (batch size, 2, min(num objects, num slots))
              containing the indices for the resulting matching.
    """

    # List of tuples of size 2 containing flat arrays
    indices = list(map(linear_sum_assignment, cost_matrix.cpu().detach().numpy()))
    indices = torch.LongTensor(np.array(indices))
    smallest_cost_matrix = torch.stack(
        [
            cost_matrix[i][indices[i, 0], indices[i, 1]]
            for i in range(cost_matrix.shape[0])
        ]
    )
    device = cost_matrix.device
    return smallest_cost_matrix.to(device)*-1, indices.to(device)


def build_grid(resolution):
    """
    Builds grid from image given resolution
    Code taken from: https://github.com/evelinehong/slot-attention-pytorch/blob/master/model.py

    Args:
        resolution: tuple containing width and height of image (width, height)
    """
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).to(device)

def construct_invertible_mlp(os_dim, zs_dim, n_layers, nonlinear=True, n_iter_cond_thresh=10000,
                             cond_thresh_ratio=0.25):

    """
     Create an (approximately) invertible mixing network based on an MLP.
     Based on the mixing code by Hyvarinen et al.
     Code taken from: https://github.com/brendel-group/cl-ica/blob/master/invertible_network_utils.py

     Args:
         os_dim: Dimensionality of the output for each rendered latent slot
         zs_dim: Dimensionality of each latent slot
         n_layers: Number of layers in the MLP.
         n_iter_cond_thresh: How many random matrices to use as a pool to find weights.
         cond_thresh_ratio: Relative threshold how much the invertibility
             (based on the condition number) can be violated in each layer.

    Returns:
        PyTorch invertible MLP mixing
     """

    layers = []
    if nonlinear:
        act_fct = torch.nn.LeakyReLU
    else:
        act_fct = torch.nn.Identity

        # Subfuction to normalize mixing matrix

    def l2_normalize(Amat, axis=0):
        # axis: 0=column-normalization, 1=row-normalization
        l2norm = np.sqrt(np.sum(Amat * Amat, axis))
        Amat = Amat / l2norm
        return Amat

    condList = np.zeros([n_iter_cond_thresh])
    for i in range(n_iter_cond_thresh):
        if i == 0:
            A = np.random.uniform(-10, 10, [os_dim, zs_dim])
        else:
            A = np.random.uniform(-10, 10, [os_dim, os_dim])
        A = l2_normalize(A, axis=0)
        condList[i] = np.linalg.cond(A)
    condList.sort()  # Ascending order
    condThresh = condList[int(n_iter_cond_thresh * cond_thresh_ratio)]

    for i in range(n_layers):

        if i == 0:
            lin_layer = nn.Linear(os_dim, zs_dim, bias=False)
        else:
            lin_layer = nn.Linear(os_dim, os_dim, bias=False)
        condA = condThresh + 1
        while condA > condThresh:
            if i == 0:
                weight_matrix = np.random.uniform(-10, 10, (os_dim, zs_dim))
            else:
                weight_matrix = np.random.uniform(-10, 10, (os_dim, os_dim))
            weight_matrix = l2_normalize(weight_matrix, axis=0)
            condA = np.linalg.cond(weight_matrix)

        lin_layer.weight.data = torch.tensor(weight_matrix, dtype=torch.float32)

        layers.append(lin_layer)

        if i < n_layers - 1:
            layers.append(act_fct())

    mixing_net = nn.Sequential(*layers)

    # fix parameters
    for p in mixing_net.parameters():
        p.requires_grad = False

    return mixing_net