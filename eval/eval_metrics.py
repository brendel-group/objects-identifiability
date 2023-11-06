import numpy as np
import torch
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import kernel_ridge
from typing import Union, Literal


def compositional_contrast(
    jac: torch.Tensor,
    slot_dim: int,
    data: Union[Literal["synth"], Literal["spriteworld"]],
) -> torch.Tensor:
    """
    Evaluates the weighted and unweighted compositional contrast values of a batch of Jacobian matrices.

    Args:
        jac: Tensor containing batch of Jacobian matrices
        slot_dim: Slot dimension of a model.
        data: Whether data is synthetic or image data.

    Returns:
        unweighted or weighted mean compositional contrast values depending on dataset used
    """
    batch_size, obs_dim, z_dim = jac.shape[0], jac.shape[1], jac.shape[2]
    num_slots = int(z_dim / slot_dim)

    jac = jac.reshape(
        batch_size * obs_dim, z_dim
    )  # batch_size*obs_dim x num_slots*slot_dim
    slot_rows = torch.stack(
        torch.split(jac, slot_dim, dim=1)
    )  # num_slots x batch_size*obs_dim x slot_dim

    # Get norms for all pixels wrt each latent slot for all samples in batch
    slot_norms = torch.norm(slot_rows, dim=2)  # num_slots x batch_size*obs_dim
    slot_norms = slot_norms.view(num_slots, batch_size, obs_dim).permute(
        1, 0, 2
    )  # batch_size x num_slots x obs_dim

    # Add small constant to reduce numerical instability
    slot_norms = slot_norms + 1e-12

    # Get mean pixel gradient norm across slots for each pixel
    slot_norms_mean = slot_norms.sum(1) / num_slots

    # Normalized slot norms based on mean gradient for each pixel
    slot_norms_norm = slot_norms / slot_norms_mean.unsqueeze(1).repeat(1, num_slots, 1)

    # Get the maximum mean gradient across all pixels
    max_norm_all = torch.max(slot_norms_mean, 1)[0]

    # Get weights for each pixel, weighting using max mean pixel gradient norm
    weights = slot_norms_mean / max_norm_all.unsqueeze(1).repeat(1, obs_dim)

    # Sum all pairwise norm products for every pixel
    comp_conts = 0
    comp_conts_norm = 0
    for i in range(num_slots):
        for j in range(i, num_slots - 1):
            comp_conts += slot_norms[:, i] * slot_norms[:, j + 1]
            comp_conts_norm += slot_norms_norm[:, i] * slot_norms_norm[:, j + 1]

    # Get mean unweighted compositional contrast
    unweight_comp_cont = comp_conts.sum(1).mean()

    # Get mean weighted compositional contrast
    weight_comp_cont = (comp_conts_norm * weights).sum(1).mean()

    if data == "synth":
        return unweight_comp_cont

    elif data == "spriteworld":
        return weight_comp_cont


def correlation(Z: torch.Tensor, hZ: torch.Tensor) -> np.ndarray:
    """
    Computes matrix of R2 scores between all inferred and ground-truth latent slots

    Args:
        Z: Tensor containing all ground-truth latents
        hZ: Tensor containing all inferred latents

    Returns:
        numpy array of R2 scores of shape: [num_slots x num_slots]
    """
    num_slots = Z.shape[1]

    # Initialize matrix of R2 scores
    corr = np.zeros((num_slots, num_slots))

    hZ = hZ.permute(1, 0, 2).cpu()
    Z = Z.permute(1, 0, 2).cpu()

    # 'hZ', 'Z' have shape: [num_slots, num_samples, slot_dim]

    # Use kernel ridge regression to predict ground-truth from inferred slots
    reg_func = lambda: kernel_ridge.KernelRidge(kernel="rbf", alpha=1.0, gamma=None)
    for i in range(num_slots):
        for j in range(num_slots):
            ZS = Z[i]
            hZS = hZ[j]

            # Standardize latents
            scaler_Z = StandardScaler()
            scaler_hZ = StandardScaler()

            z_train, z_eval = np.split(scaler_Z.fit_transform(ZS), [int(0.8 * len(ZS))])
            hz_train, hz_eval = np.split(
                scaler_hZ.fit_transform(hZS), [int(0.8 * len(hZS))]
            )

            # Fit KRR model
            reg_model = reg_func()
            reg_model.fit(hz_train, z_train)
            hz_pred_val = reg_model.predict(hz_eval)

            # Populate correlation matrix
            corr[i, j] = sklearn.metrics.r2_score(z_eval, hz_pred_val)

    return corr
