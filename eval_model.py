import torch
import utils
from eval.model_evaluation import evaluate_model
from eval.eval_metrics import compositional_contrast, correlation


def eval_model(args, model, data_loader):
    """
    Evaluates an object-centric model

    Args:
        args: Command line arguments from train_model.py specifying: number of slots, inferred slot dimension, dataset type
        model: object-centric PyTorch model
        data_loader: PyTorch dataloader for validation set

    Returns:
        reconstruction loss, compositional contrast, and slot identifiability score computed on validation set
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # eval step
    model.eval()
    with torch.no_grad():
        run_recon_loss, run_c_comp = 0, 0
        b_it = 0
        for x, z in data_loader:
            x = x.to(device)

            if args.encoder == "monet" or args.decoder == "monet":
                zh, xh, _ = model(x)
            else:
                zh, xh = model(x)

            # get recon_loss
            run_recon_loss += ((x - xh).square().mean()).item()

            # get c_comp
            jacobian = torch.vmap(torch.func.jacfwd(model.decoder))(zh.flatten(1))
            if args.data == "spriteworld":
                jacobian = jacobian.flatten(1, 4)
            run_c_comp += compositional_contrast(jacobian, args.inf_slot_dim, args.data).item()


            # save latents
            if b_it == 0:
                Z = z
                Zh = zh
            else:
                Z = torch.cat((Z, z))
                Zh = torch.cat((Zh, zh))

            b_it += 1

    # get slot identifiability score
    if args.data == "synth":

        # get matrix of R2 scores
        corr_mat = correlation(Z, Zh)
        corr_mat = torch.nn.functional.relu(torch.from_numpy(corr_mat))

        # resolve permutation
        _, inds = utils.hungarian_algorithm(corr_mat.view(1, args.num_slots, args.num_slots)*-1)
        perm_corr_mat = corr_mat[:, inds[0][1]]

        # get mean R2 on-diagonal
        r2_on = perm_corr_mat.diag()

        # get max R2 off-diagonal
        r2_off = torch.max((perm_corr_mat - torch.diag(r2_on)), 1)[0]

        # compute slot identifiability score
        SIS = (r2_on - r2_off).mean().item()

    elif args.data == "spriteworld":
        z = Z.reshape(len(Z), args.num_slots, -1).to(device)
        z_pred = Zh.reshape(len(Zh), args.num_slots, -1).to(device)
        SIS_metrics = evaluate_model(
            z_pred=z_pred, z=z, categorical_dimensions=[4], verbose=2, z_mask_values=0
        )
        SIS = SIS_metrics[0]

    return (run_recon_loss / b_it), (run_c_comp / b_it), SIS
