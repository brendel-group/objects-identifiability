import torch
import utils
from eval.model_evaluation import evaluate_model
from eval.eval_metrics import compositional_contrast, correlation


def eval_model(args, model, data_loader):
    """
    Evaluates an object-centric model

    Args:
        args: Command line arguments from train_model.py specifying: number of slots, inferred slot dimension, dataset type.
        model: object-centric PyTorch model.
        data_loader: PyTorch dataloader for validation set.

    Returns:
        reconstruction loss, compositional contrast, and slot identifiability score computed on validation set.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Eval step by computing the inferred latents and the compositional contrast
    model.eval()
    with torch.no_grad():
        raise NotImplementedError()

    # Get slot identifiability score
    if args.data == "synth":
        raise NotImplementedError()

    elif args.data == "spriteworld":
        z = Z.reshape(len(Z), args.num_slots, -1).to(device)
        z_pred = Zh.reshape(len(Zh), args.num_slots, -1).to(device)
        SIS = evaluate_model(
            z_pred=z_pred, z=z, categorical_dimensions=[4], verbose=2, z_mask_values=0
        )

    return (run_recon_loss / b_it), (run_c_comp / b_it), SIS
