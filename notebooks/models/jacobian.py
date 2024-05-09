import torch
from torch.autograd import grad

def compute_jacobian(model, z_batch):
    """
    Compute the Jacobian of the decoder wrt a batch of points in the latent space using an efficient broadcasting approach.
    :param model: The VAE model.
    :param z_batch: A batch of points in the latent space (tensor).
    :return: A batch of Jacobian matrices.
    """
    # z_batch = z_batch.clone().detach().requires_grad_(True)
    z_batch.requires_grad_(True)
    # model.no_grad()
    output = model.decoder(z_batch)
    batch_size, output_dim, latent_dim = *output.shape, z_batch.shape[-1]

    # Use autograd's grad function to get gradients for each output dimension
    jacobian = torch.zeros(batch_size, output_dim, latent_dim).to(z_batch.device)
    for i in range(output_dim):
        grad_outputs = torch.zeros(batch_size, output_dim).to(z_batch.device)
        grad_outputs[:, i] = 1.0
        gradients = grad(outputs=output, inputs=z_batch, grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
        jacobian[:, i, :] = gradients

    return jacobian

def gram_schmidt_batched(A):
    """
    Perform the Gram-Schmidt process on a batch of matrices.
    :param A: Batched tensor of shape (n_batch, m, n).
    :return: Orthonormalized batched tensor.
    """
    n_batch, m, n = A.shape
    Q = torch.zeros_like(A)
    
    for j in range(n):
        # Start with the original vector
        v = A[:, :, j]
        
        # Orthogonalize v w.r.t. the previous vectors in Q
        for k in range(j):
            v = v - (torch.sum(v * Q[:, :, k], dim=1, keepdim=True) * Q[:, :, k])
        
        # Normalize v
        v = v / torch.linalg.norm(v, dim=1, keepdim=True)
        
        # Store in Q
        Q[:, :, j] = v

    return Q

# def compute_cristoffel_symbols(model, z_batch):
#     jac = compute_jacobian(model, z_batch)
#     # Orthogonalize the Jacobian tensor using QR decomposition
#     Q, _ = torch.qr(jac.transpose(1, 2))
#     orthonormal_basis = Q.transpose(1, 2)
#     # Compute the metric tensor g_ij using the orthonormal basis
#     metric_tensor = torch.einsum('bik,bjk->bij', orthonormal_basis, orthonormal_basis)
#     # Reshape the metric_tensor to treat each slice as a separate item in the batch dimension
#     latent_dim = z_batch.size(1)
#     reshaped_metric_tensor = metric_tensor.view(-1, latent_dim)

#     # Compute the gradient for the entire reshaped tensor
#     metric_derivatives_reshaped = grad(reshaped_metric_tensor.sum(), z_batch, 
#                                                     retain_graph=True, create_graph=True)[0]

#     # Reshape the metric_derivatives back to the original shape
#     metric_derivatives = metric_derivatives_reshaped.view(-1, latent_dim, latent_dim)

#     # Compute the inverse of the metric tensor g^kl
#     # metric_tensor_inverse = torch.linalg.inv(metric_tensor)
#     metric_tensor_inverse = torch.inverse(metric_tensor)

#     # Compute the Christoffel symbols
#     christoffel_symbols = 0.5 * (
#         torch.einsum('bkl,bij->bkl', metric_tensor_inverse, metric_derivatives) +
#         torch.einsum('bkl,bji->bkl', metric_tensor_inverse, metric_derivatives) -
#         torch.einsum('bkl,bij->bkl', metric_tensor_inverse, metric_tensor)
#     )
#     return christoffel_symbols