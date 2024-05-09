from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt
from models.jacobian import compute_jacobian, gram_schmidt_batched
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Create a class for 3D arrows
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

def visualize_jacobian_at_z0(model, z0, background_data):
    # Decode z0 to get x0
    x0 = model.decoder(z0).squeeze()
    x0_np = x0.detach().numpy()
    
    # Compute the Jacobian for z0
    jacobian = compute_jacobian(model, z0).squeeze()
    jacobian_np = jacobian.detach().numpy()
    
    # Set up a 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(background_data[:, 0], background_data[:, 1], background_data[:, 2], c='b', marker='o', alpha=0.3)
    ax.scatter(x0_np[0], x0_np[1], x0_np[2], c='red', s=100, label='Decoded x0 from z0')

    # Plot the Jacobian vectors
    for i in range(jacobian_np.shape[1]):
        arrow = Arrow3D([x0_np[0], x0_np[0] + jacobian_np[0, i]],
                        [x0_np[1], x0_np[1] + jacobian_np[1, i]],
                        [x0_np[2], x0_np[2] + jacobian_np[2, i]],
                        mutation_scale=15, lw=2, arrowstyle="-|>", color="k")
        ax.add_artist(arrow)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])
    ax.set_title('Visualization of Jacobian Vectors at x0')
    ax.legend()
    plt.show()


def visualize_jacobian_plane_at_z0(model, z0):
    # Decode z0 to get x0
    x0 = model.decoder(z0).squeeze()
    x0_np = x0.detach().numpy()
    
    # Compute the Jacobian for z0
    jacobian = compute_jacobian(model, z0).squeeze()
    jacobian_np = jacobian.detach().numpy()
    
    # Get the two basis vectors from the Jacobian
    v1 = jacobian_np[:, 0]
    v2 = jacobian_np[:, 1]
    
    # Compute four corners of the plane patch
    scale = 1  # scaling factor to control the size of the plane patch
    corner1 = x0_np + scale * v1 + scale * v2
    corner2 = x0_np - scale * v1 + scale * v2
    corner3 = x0_np - scale * v1 - scale * v2
    corner4 = x0_np + scale * v1 - scale * v2
    
    # Set up a 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(background_data[:, 0], background_data[:, 1], background_data[:, 2], c='b', marker='o', alpha=0.3)
    ax.scatter(x0_np[0], x0_np[1], x0_np[2], c='red', s=100, label='Decoded x0 from z0')
    
    # Plot the Jacobian vectors
    for i in range(jacobian_np.shape[1]):
        arrow = Arrow3D([x0_np[0], x0_np[0] + jacobian_np[0, i]],
                        [x0_np[1], x0_np[1] + jacobian_np[1, i]],
                        [x0_np[2], x0_np[2] + jacobian_np[2, i]],
                        mutation_scale=15, lw=2, arrowstyle="-|>", color="k")
        ax.add_artist(arrow)
    
    # Plot the plane spanned by the Jacobian vectors
    vertices = [list(corner1), list(corner2), list(corner3), list(corner4)]
    ax.add_collection3d(Poly3DCollection([vertices], alpha=0.5, facecolors='cyan'))
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])
    ax.set_title('Visualization of Jacobian Vectors and Plane at x0')
    ax.legend()
    plt.show()



def visualize_jacobian_plane_at_z0_schmidt(model, z0, background_data):
    # Decode z0 to get x0
    x0 = model.decoder(z0).squeeze()
    x0_np = x0.detach().numpy()
    
    # Compute the Jacobian for z0
    jacobian = compute_jacobian(model, z0)
    jacobian_np = gram_schmidt_batched(jacobian).squeeze()
    jacobian_np = jacobian_np.detach().numpy()
    
    # Get the two basis vectors from the Jacobian
    v1 = jacobian_np[:, 0]
    v2 = jacobian_np[:, 1]
    
    # Compute four corners of the plane patch
    scale = 1  # scaling factor to control the size of the plane patch
    corner1 = x0_np + scale * v1 + scale * v2
    corner2 = x0_np - scale * v1 + scale * v2
    corner3 = x0_np - scale * v1 - scale * v2
    corner4 = x0_np + scale * v1 - scale * v2
    
    # Set up a 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(background_data[:, 0], background_data[:, 1], background_data[:, 2], c='b', marker='o', alpha=0.3)
    ax.scatter(x0_np[0], x0_np[1], x0_np[2], c='red', s=100, label='Decoded x0 from z0')
    
    # Plot the Jacobian vectors
    for i in range(jacobian_np.shape[1]):
        arrow = Arrow3D([x0_np[0], x0_np[0] + jacobian_np[0, i]],
                        [x0_np[1], x0_np[1] + jacobian_np[1, i]],
                        [x0_np[2], x0_np[2] + jacobian_np[2, i]],
                        mutation_scale=15, lw=2, arrowstyle="-|>", color="k")
        ax.add_artist(arrow)
    
    # Plot the plane spanned by the Jacobian vectors
    vertices = [list(corner1), list(corner2), list(corner3), list(corner4)]
    ax.add_collection3d(Poly3DCollection([vertices], alpha=0.5, facecolors='cyan'))
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])
    ax.set_title('Visualization of Jacobian Vectors and Plane at x0')
    ax.legend()
    plt.show()