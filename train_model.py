import random
import torch
import numpy as np
import argparse
import warnings
from eval.eval_metrics import compositional_contrast
from eval_model import eval_model
import utils
import wandb

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def train_model(args):
    """
    Trains an object-centric model
    Prints evaluation metrics every args.eval_iter iterations

    Args:
        args: Command line arguments specifying training setup
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # fix random seed
    seed = random.randint(0, 10000)
    utils.set_seed(seed)

    # get directory to save model logs
    model_dir = utils.setup_direcs(args, seed)

    # create model
    model = utils.get_model(args)

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # load data
    train_loader, val_loader = utils.get_data(args)

    # train loop
    b_it, glob_it = 0, 0
    run_recon_loss = 0.0
    while glob_it < args.num_iters:
        model.train()
        x, _ = next(iter(train_loader))
        b_it += 1
        optimizer.zero_grad()
        x = x.to(device)

        if args.encoder == "monet" or args.decoder == "monet":
            zh, xh, total_loss = model(x)
        else:
            zh, xh = model(x)
            total_loss = None

        # recon loss
        recon_loss = (x - xh).square().mean()
        run_recon_loss += recon_loss.item()

        # c_comp
        if args.lam > 0:
            jacobian = torch.vmap(torch.func.jacfwd(model.decoder))(zh.flatten(1))
            c_comp = compositional_contrast(jacobian, args.inf_slot_dim, args.data)
        else:
            with torch.no_grad():
                c_comp = torch.Tensor([0.0]).to(device)

        # total loss
        if total_loss == None:
            total_loss = recon_loss + args.lam * c_comp

        total_loss.backward()
        optimizer.step()
        glob_it += 1

        # lr decay
        if args.data == "spriteworld":
            decay_rate = 0.5
            decay_steps = 100000
            optimizer.param_groups[0]["lr"] = args.lr * (
                decay_rate ** (glob_it / decay_steps)
            )

        elif args.data == "synth":
            if glob_it == int(args.num_iters * 0.5):
                optimizer.param_groups[0]["lr"] = args.lr / 10

        # save model
        if glob_it % 3000 == 0:
            torch.save(
                model.state_dict(),
                model_dir + "_iter_" + str(glob_it) + "_model_state_dict.pt",
            )

        # eval model
        if glob_it == 1 or glob_it % args.eval_iter == 0 or glob_it == args.num_iters:
            train_recon = run_recon_loss / b_it
            val_recon, val_c_comp, val_sis = eval_model(args, model, val_loader)
            b_it = 0
            run_recon_loss = 0.0

            print(
                "Iteration: ",
                glob_it,
                "Train Recon: ",
                train_recon,
                "Val Recon: ",
                val_recon,
                "Val C_comp: ",
                val_c_comp,
                "Val SIS: ",
                val_sis,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        help="Specifies whether to use image data or not",
        default="synth",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        help="Specifies encoder to be used for image experiments",
        default="MLP",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        help="Specifies decoder to be used for image experiments",
        default="MLP",
    )
    parser.add_argument(
        "--num_slots",
        type=int,
        help="Specifies number of slots in ground-truth and inference model",
        default="2",
    )
    parser.add_argument(
        "--inf_slot_dim",
        type=int,
        help="Specifies slot dimension in inference model",
        default="3",
    )
    parser.add_argument(
        "--gt_slot_dim",
        type=int,
        help="Specifies slot dimension in ground-truth model",
        default="3",
    )
    parser.add_argument(
        "--lam",
        help="Specifies the coefficient on the compositional contrast",
        type=float,
        default="0",
    )
    parser.add_argument("--batch_size", type=int, default="64")
    parser.add_argument("--lr", type=float, default="4e-4")
    parser.add_argument(
        "--num_iters",
        help="Specifies the number of training iterations",
        type=int,
        default="200000",
    )
    parser.add_argument(
        "--eval_iter",
        help="Evaluation metrics computed and printed every number of iterations given by arg",
        type=int,
        default="5000",
    )
    parser.add_argument(
        "--nobs", help="Size of dataset for non-image data", type=int, default="80000"
    )
    parser.add_argument(
        "--slot_x_dim",
        help="Dimension of slot output for ground-truth model for non-image data",
        type=int,
        default="20",
    )
    parser.add_argument(
        "--dependent",
        help="0 if slots are sampled independently and 1 for dependently for non-image data",
        type=int,
        default="0",
    )

    args = parser.parse_args()

    if args.data == "synth":
        args.encoder = "MLP"
        args.decoder = "MLP"

    if args.data == "spriteworld":
        args.gt_slot_dim = 5
        args.lam = 0

    train_model(args)
