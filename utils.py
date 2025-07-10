import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from torch.utils import data
from torch.autograd import Variable
import torch
import os
import pickle
from PIL import Image
from torchvision import transforms
import binvox_rw

def getVolumeFromBinvox(path):
    with open(path, 'rb') as file:
        data = np.int32(binvox_rw.read_as_3d_array(file).data)
    return data

def SavePloat_Voxels(voxels, path, iteration):
    voxels = voxels[:8].__ge__(0.5)
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(voxels):
        x, y, z = sample.nonzero()
        ax = plt.subplot(gs[i], projection='3d')
        ax.scatter(x, y, z, zdir='z', c='red')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
    #plt.show()
    plt.savefig(path + '/{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
    plt.close()

    with open(path + '/{}.pkl'.format(str(iteration).zfill(3)), "wb") as f:
        pickle.dump(voxels, f, protocol=pickle.HIGHEST_PROTOCOL)


def make_hyparam_string(hyparam_dict):
    str_result = ""
    for i in hyparam_dict.keys():
        str_result = str_result + str(i) + "=" + str(hyparam_dict[i]) + "_"
    return str_result[:-1]

class ShapeNetMultiviewDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader"""

    def __init__(self, root, args):
        """Set the path for Data.

        Args:
            root: image directory.
            transform: Tensor transformer.
        """
        self.root = root
        self.listdir = os.listdir(self.root)
        self.args = args
        self.img_size = args.image_size
        self.p = transforms.Compose([transforms.Resize((self.img_size, self.img_size))])

    def __getitem__(self, index):
        model_3d_file = [name for name in self.listdir if name.endswith('.' + "binvox")][index]

        model_2d_files = [name for name in self.listdir if name.startswith(model_3d_file[:-7]) and name.endswith(".png")][:self.args.num_views]
        volume = np.asarray(getVolumeFromBinvox(self.root + model_3d_file), dtype=np.float32)
        images = [torch.FloatTensor(np.asarray(self.p(Image.open(self.root +x ))).copy()) for x in model_2d_files]  # copy() ekledik
        return (images, torch.FloatTensor(volume))

    def __len__(self):
        return len( [name for name in self.listdir if name.endswith('.' + "binvox")])


def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return Variable(x)

def generateZ(args):

    if args.z_dis == "norm":
        Z = var_or_cuda(torch.Tensor(args.batch_size, args.z_size).normal_(0, 0.33))
    elif args.z_dis == "uni":
        Z = var_or_cuda(torch.randn(args.batch_size, args.z_size))
    else:
        print("z_dist is not normal or uniform")

    return Z

########################## Pickle helper ###############################


def read_pickle(path, G, G_solver, D_, D_solver,E_=None,E_solver = None ):
    try:

        files = os.listdir(path)
        file_list = [int(file.split('_')[-1].split('.')[0]) for file in files]
        file_list.sort()
        recent_iter = str(file_list[-1])
        print(recent_iter, path)

        with open(path + "/G_" + recent_iter + ".pkl", "rb") as f:
            G.load_state_dict(torch.load(f))
        with open(path + "/G_optim_" + recent_iter + ".pkl", "rb") as f:
            G_solver.load_state_dict(torch.load(f))
        with open(path + "/D_" + recent_iter + ".pkl", "rb") as f:
            D_.load_state_dict(torch.load(f))
        with open(path + "/D_optim_" + recent_iter + ".pkl", "rb") as f:
            D_solver.load_state_dict(torch.load(f))
        if E_ is not None:
            with open(path + "/E_" + recent_iter + ".pkl", "rb") as f:
                E_.load_state_dict(torch.load(f))
            with open(path + "/E_optim_" + recent_iter + ".pkl", "rb") as f:
                E_solver.load_state_dict(torch.load(f))


    except Exception as e:
        print("fail try read_pickle", e)



def save_new_pickle(path, iteration, G, G_solver, D_, D_solver, E_=None,E_solver = None):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + "/G_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(G.state_dict(), f)
    with open(path + "/G_optim_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(G_solver.state_dict(), f)
    with open(path + "/D_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(D_.state_dict(), f)
    with open(path + "/D_optim_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(D_solver.state_dict(), f)
    if E_ is not None:
        with open(path + "/E_" + str(iteration) + ".pkl", "wb") as f:
            torch.save(E_.state_dict(), f)
        with open(path + "/E_optim_" + str(iteration) + ".pkl", "wb") as f:
            torch.save(E_solver.state_dict(), f)

def calculate_iou(pred, target, threshold=0.5):
    """
    Calculate Intersection over Union (IoU) between predicted and target voxels.
    Computes IoU per sample in the batch and returns the average.
    Args:
        pred: predicted voxels (torch.Tensor)
        target: ground truth voxels (torch.Tensor)
        threshold: threshold for binarization
    Returns:
        iou score (float)
    """
    pred = (pred >= threshold).float()
    target = (target >= threshold).float()
    
    # Calculate intersection and union per sample
    # Keep batch dimension (dim=0)
    intersection = torch.sum(pred * target, dim=(1, 2, 3, 4))
    union = torch.sum(pred, dim=(1, 2, 3, 4)) + torch.sum(target, dim=(1, 2, 3, 4)) - intersection
    
    # Compute IoU per sample, handle division by zero
    iou_per_sample = intersection / (union + 1e-6)
    
    # Return the average IoU over the batch
    return torch.mean(iou_per_sample).item()

# Try a more standard 3D viewing angle first
def save_comparison_plot(input_image, generated_voxels, true_voxels, path, iteration, threshold=0.5, elev=30, azim=45):
    """Generates and saves a 4-column comparison plot with print-friendly visuals."""
    
    # Ensure tensors are on CPU and converted to numpy
    img_np = input_image.detach().cpu().numpy()
    if img_np.shape[0] in [3, 4]: # Check if channels-first
        img_np = np.transpose(img_np, (1, 2, 0))
    # Handle RGBA or RGB
    if img_np.shape[2] == 4:
        img_np = img_np[:, :, :3] # Take only RGB for display
    # Normalize if needed
    if img_np.max() > 1.0:
        img_np = img_np / 255.0
    img_np = np.clip(img_np, 0, 1) # Ensure valid range

    # Voxels: Binarize
    gen_vox_np = (generated_voxels.detach().cpu().numpy().squeeze() >= threshold)
    true_vox_np = (true_voxels.detach().cpu().numpy().squeeze() >= threshold)

    # Transpose for consistent orientation
    gen_vox_np = np.transpose(gen_vox_np, (0, 2, 1))
    true_vox_np = np.transpose(true_vox_np, (0, 2, 1))

    # Calculate intersection (matching voxels)
    intersection_vox_np = np.logical_and(gen_vox_np, true_vox_np)

    # Define publication-friendly colors (optimized for printing)
    color_generated = '#FF3333'      # Bright red
    color_true = '#3333FF'           # Bright blue
    color_intersection = '#33AA33'   # Darker green for better print visibility
    edge_color = '#555555'           # Medium gray edges
    alpha_solid = 1.0                # Fully opaque for better print clarity
    alpha_overlay = 0.8              # Higher alpha for print clarity

    # Common function to set up axes for 3D plots
    def setup_3d_axes_common(ax, with_labels=True, title=''):
        # Set equal scaling for all axes
        dims = gen_vox_np.shape
        max_dim = max(dims)
        ax.set_xlim(0, max_dim)
        ax.set_ylim(0, max_dim)
        ax.set_zlim(0, max_dim)
        
        # Set the aspect ratio to be equal
        if hasattr(ax, 'set_box_aspect'):
            ax.set_box_aspect([1, 1, 1])
        
        # View angle with slightly offset for better depth perception
        ax.view_init(elev=elev, azim=azim)
        
        # Only set labels and titles if requested
        if with_labels:
            if title:
                ax.set_title(title, color='black', fontsize=14, fontweight='bold')
        
        # Hide axis ticks and labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Enhance pane colors for better depth perception
        ax.xaxis.set_pane_color((0.95, 0.95, 0.98, 1.0))
        ax.yaxis.set_pane_color((0.93, 0.93, 0.96, 1.0))
        ax.zaxis.set_pane_color((0.91, 0.91, 0.94, 1.0))
        
        # Add subtle grid for better spatial reference
        ax.xaxis._axinfo["grid"]['color'] = (0.7, 0.7, 0.7, 0.5)
        ax.yaxis._axinfo["grid"]['color'] = (0.7, 0.7, 0.7, 0.5)
        ax.zaxis._axinfo["grid"]['color'] = (0.7, 0.7, 0.7, 0.5)
    
    # Function to create the comparison figure with labels and titles
    def create_labeled_figure():
        plt.style.use('default')
        fig = plt.figure(figsize=(22, 7))
        gs = gridspec.GridSpec(1, 4, width_ratios=[0.8, 1, 1, 1.2])
        gs.update(wspace=0.15, hspace=0)
        fig.patch.set_facecolor('white')
        
        # 1. Input Image with enhanced presentation
        ax1 = plt.subplot(gs[0])
        ax1.imshow(img_np)
        ax1.set_title('Input View', color='black', fontsize=14, fontweight='bold')
        ax1.set_facecolor('#F8F8F8')
        for spine in ax1.spines.values():
            spine.set_color('#CCCCCC')
        ax1.axis('off')

        # 2. Generated Voxels with publication-friendly visuals
        ax2 = plt.subplot(gs[1], projection='3d')
        ax2.voxels(gen_vox_np, facecolors=color_generated, edgecolor=edge_color, alpha=alpha_solid)
        setup_3d_axes_common(ax2, title='Generated Model')

        # 3. True Voxels with publication-friendly visuals
        ax3 = plt.subplot(gs[2], projection='3d')
        ax3.voxels(true_vox_np, facecolors=color_true, edgecolor=edge_color, alpha=alpha_solid)
        setup_3d_axes_common(ax3, title='Ground Truth')

        # 4. Overlay Comparison with color-coded matches
        ax4 = plt.subplot(gs[3], projection='3d')
        
        # True only voxels
        true_only = np.logical_and(true_vox_np, np.logical_not(intersection_vox_np))
        ax4.voxels(true_only, facecolors=color_true, edgecolor=edge_color, alpha=alpha_overlay)
        
        # Generated only voxels
        gen_only = np.logical_and(gen_vox_np, np.logical_not(intersection_vox_np))
        ax4.voxels(gen_only, facecolors=color_generated, edgecolor=edge_color, alpha=alpha_overlay)
        
        # Intersection (matching) voxels
        ax4.voxels(intersection_vox_np, facecolors=color_intersection, edgecolor=edge_color, alpha=alpha_solid)
        
        # Set up the axes first
        setup_3d_axes_common(ax4)
        
        # Then set the title separately
        ax4.set_title('Overlay Comparison\nGreen: Match  Blue: GT Only  Red: Gen Only', 
                    color='black', fontsize=14, fontweight='bold')
        
        # Calculate IoU for annotation
        iou = np.sum(intersection_vox_np) / (np.sum(gen_vox_np) + np.sum(true_vox_np) - np.sum(intersection_vox_np))
        ax4.text2D(0.05, 0.05, f"IoU: {iou:.3f}", transform=ax4.transAxes, 
                color='black', fontsize=12, bbox=dict(facecolor='white', edgecolor='#AAAAAA', alpha=0.9))

        plt.tight_layout()
        return fig
    
    # Function to create the comparison figure without any labels or titles
    def create_clean_figure():
        plt.style.use('default')
        fig = plt.figure(figsize=(22, 7))
        gs = gridspec.GridSpec(1, 4, width_ratios=[0.8, 1, 1, 1.2])
        gs.update(wspace=0.15, hspace=0)
        fig.patch.set_facecolor('white')
        
        # 1. Input Image
        ax1 = plt.subplot(gs[0])
        ax1.imshow(img_np)
        ax1.set_facecolor('white')
        ax1.axis('off')
        # Remove all spines
        for spine in ax1.spines.values():
            spine.set_visible(False)

        # 2. Generated Voxels
        ax2 = plt.subplot(gs[1], projection='3d')
        ax2.voxels(gen_vox_np, facecolors=color_generated, edgecolor=edge_color, alpha=alpha_solid)
        setup_3d_axes_common(ax2, with_labels=False)
        
        # Remove all axis lines, ticks, etc.
        ax2.set_axis_off()

        # 3. True Voxels
        ax3 = plt.subplot(gs[2], projection='3d')
        ax3.voxels(true_vox_np, facecolors=color_true, edgecolor=edge_color, alpha=alpha_solid)
        setup_3d_axes_common(ax3, with_labels=False)
        
        # Remove all axis lines, ticks, etc.
        ax3.set_axis_off()

        # 4. Overlay Comparison
        ax4 = plt.subplot(gs[3], projection='3d')
        
        # True only voxels
        true_only = np.logical_and(true_vox_np, np.logical_not(intersection_vox_np))
        ax4.voxels(true_only, facecolors=color_true, edgecolor=edge_color, alpha=alpha_overlay)
        
        # Generated only voxels
        gen_only = np.logical_and(gen_vox_np, np.logical_not(intersection_vox_np))
        ax4.voxels(gen_only, facecolors=color_generated, edgecolor=edge_color, alpha=alpha_overlay)
        
        # Intersection (matching) voxels
        ax4.voxels(intersection_vox_np, facecolors=color_intersection, edgecolor=edge_color, alpha=alpha_solid)
        
        setup_3d_axes_common(ax4, with_labels=False)
        
        # Remove all axis lines, ticks, etc.
        ax4.set_axis_off()

        plt.tight_layout()
        return fig
    
    # Save the labeled version
    fig_labeled = create_labeled_figure()
    fig_labeled.savefig(path + '/comparison_{}.png'.format(str(iteration).zfill(3)), 
                bbox_inches='tight', dpi=300, facecolor='white')
    fig_labeled.savefig(path + '/comparison_{}.pdf'.format(str(iteration).zfill(3)), 
                bbox_inches='tight', format='pdf', facecolor='white')
    plt.close(fig_labeled)
    
    # Save the clean version without labels
    fig_clean = create_clean_figure()
    fig_clean.savefig(path + '/comparison_{}_clean.png'.format(str(iteration).zfill(3)), 
                bbox_inches='tight', dpi=300, facecolor='white')
    fig_clean.savefig(path + '/comparison_{}_clean.pdf'.format(str(iteration).zfill(3)), 
                bbox_inches='tight', format='pdf', facecolor='white')
    plt.close(fig_clean)

def voxel_to_obj(voxel_array, threshold=0.5, scale=1.0):
    """
    Convert a voxel array to OBJ file format string.
    Args:
        voxel_array: numpy 3D array representing voxels (values >= threshold will be converted to cubes)
        threshold: values >= threshold will be treated as solid voxels
        scale: scaling factor for vertex coordinates (default 1.0)
    Returns:
        String containing OBJ file content
    """
    # Ensure voxel array is binary
    voxels = (voxel_array >= threshold)
    
    vertices = []
    faces = []
    vertex_count = 0
    
    # Define the 8 corners of a unit cube
    cube_corners = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ]
    
    # Define the 6 faces of a cube (each face is a quad: 4 indices to corners)
    cube_faces = [
        [0, 1, 2, 3],  # bottom face (z=0)
        [4, 5, 6, 7],  # top face (z=1)
        [0, 1, 5, 4],  # front face (y=0)
        [2, 3, 7, 6],  # back face (y=1)
        [0, 3, 7, 4],  # left face (x=0)
        [1, 2, 6, 5]   # right face (x=1)
    ]
    
    # Iterate through all voxels
    for z in range(voxels.shape[0]):
        for y in range(voxels.shape[1]):
            for x in range(voxels.shape[2]):
                if voxels[z, y, x]:
                    # Add the 8 corners of this voxel
                    for corner in cube_corners:
                        vertices.append([
                            (x + corner[0]) * scale,
                            (y + corner[1]) * scale,
                            (z + corner[2]) * scale
                        ])
                    
                    # Add the 6 faces of this voxel (each face has 4 vertices)
                    for face in cube_faces:
                        # OBJ format uses 1-based indexing
                        faces.append([
                            vertex_count + face[0] + 1, 
                            vertex_count + face[1] + 1, 
                            vertex_count + face[2] + 1, 
                            vertex_count + face[3] + 1
                        ])
                    
                    # Increment vertex count for next voxel
                    vertex_count += 8
    
    # Construct the OBJ file content
    obj_content = "# Voxel model converted to OBJ\n"
    obj_content += "# Vertices: {}\n".format(len(vertices))
    obj_content += "# Faces: {}\n\n".format(len(faces))
    
    # Add vertices
    for vertex in vertices:
        obj_content += "v {} {} {}\n".format(vertex[0], vertex[1], vertex[2])
    
    # Add faces
    for face in faces:
        obj_content += "f {} {} {} {}\n".format(face[0], face[1], face[2], face[3])
    
    return obj_content

def save_image_copy(image_tensor, save_path, filename):
    """
    Save a copy of an image tensor to the specified path.
    
    Args:
        image_tensor: The tensor representing the image
        save_path: Directory where to save the image
        filename: Filename to use for saving
    """
    # Ensure the directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Convert tensor to numpy and ensure proper format
    img_np = image_tensor.detach().cpu().numpy()
    if img_np.shape[0] in [3, 4]:  # If channels-first format
        img_np = np.transpose(img_np, (1, 2, 0))
    
    # Handle RGBA or RGB
    if img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]  # Take only RGB channels
    
    # Normalize values if needed
    if img_np.max() > 1.0:
        img_np = img_np / 255.0
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    
    # Save using PIL
    from PIL import Image
    Image.fromarray(img_np).save(os.path.join(save_path, filename))

def voxel_to_points(voxel_array, threshold=0.5):
    """
    Convert a voxel grid to a surface point cloud.
    Only extracts points at the surface (voxels with at least one empty neighbor).
    
    Args:
        voxel_array: 3D numpy array of voxel occupancy values
        threshold: Threshold for considering a voxel as occupied
    
    Returns:
        points: Nx3 array of surface point coordinates
    """
    import numpy as np
    
    # Binarize the voxel grid
    binary_voxels = (voxel_array >= threshold)
    
    # Find surface voxels (those that have at least one empty neighbor)
    # Pad with zeros to handle boundary voxels
    padded = np.pad(binary_voxels, ((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)
    
    points = []
    # Check each voxel
    for z in range(1, padded.shape[0] - 1):
        for y in range(1, padded.shape[1] - 1):
            for x in range(1, padded.shape[2] - 1):
                # If the voxel is occupied
                if padded[z, y, x]:
                    # Check its 6 direct neighbors
                    neighbors = [
                        padded[z-1, y, x], padded[z+1, y, x],
                        padded[z, y-1, x], padded[z, y+1, x],
                        padded[z, y, x-1], padded[z, y, x+1]
                    ]
                    # If at least one neighbor is empty, this is a surface voxel
                    if not all(neighbors):
                        # Add coordinates, adjusted for padding
                        points.append([x-1, y-1, z-1])
    
    return np.array(points).astype(np.float32)

def chamfer_distance_numpy(points1, points2):
    """
    Compute Chamfer Distance between two point clouds using NumPy.
    
    Args:
        points1: Nx3 numpy array of points
        points2: Mx3 numpy array of points
    
    Returns:
        chamfer_distance: Scalar Chamfer Distance value
    """
    import numpy as np
    from scipy.spatial import distance_matrix
    
    # Ensure we have points to compare
    if len(points1) == 0 or len(points2) == 0:
        return float('inf')
    
    # Calculate pairwise distances
    dist_matrix = distance_matrix(points1, points2)
    
    # For each point in points1, find the minimum distance to points2
    min_dist_1to2 = np.min(dist_matrix, axis=1)
    
    # For each point in points2, find the minimum distance to points1
    min_dist_2to1 = np.min(dist_matrix, axis=0)
    
    # Chamfer distance is the sum of the averages
    chamfer_dist = np.mean(min_dist_1to2) + np.mean(min_dist_2to1)
    
    return chamfer_dist

def calculate_chamfer_distance_gpu(pred, target, threshold=0.5, samples=1000):
    """
    GPU-accelerated Chamfer Distance calculation between predicted and target voxels.
    Up to 10-20x faster than the CPU version.
    
    Args:
        pred: predicted voxels (torch.Tensor) - already on GPU
        target: ground truth voxels (torch.Tensor) - already on GPU
        threshold: threshold for binarization
        samples: max number of points to sample for memory efficiency
    
    Returns:
        cd_score: average chamfer distance over the batch
    """
    batch_size = pred.shape[0]
    cd_scores = []
    
    # Run on same device as input tensors
    device = pred.device
    
    for i in range(batch_size):
        # Binarize voxels
        pred_binary = (pred[i, 0] >= threshold)
        target_binary = (target[i, 0] >= threshold)
        
        # Get surface voxel coordinates
        pred_indices = torch.nonzero(pred_binary, as_tuple=False).float()
        target_indices = torch.nonzero(target_binary, as_tuple=False).float()
        
        # Skip if either point cloud is empty
        if len(pred_indices) == 0 or len(target_indices) == 0:
            continue
            
        # Sample points if there are too many
        if len(pred_indices) > samples:
            perm = torch.randperm(len(pred_indices), device=device)
            pred_indices = pred_indices[perm[:samples]]
            
        if len(target_indices) > samples:
            perm = torch.randperm(len(target_indices), device=device)
            target_indices = target_indices[perm[:samples]]
        
        # Calculate pair-wise squared distances efficiently with PyTorch operations
        # Using broadcasting for batch computation
        pred_squared = torch.sum(pred_indices**2, dim=1, keepdim=True)  # [m,1]
        target_squared = torch.sum(target_indices**2, dim=1, keepdim=True)  # [n,1]
        
        # pred_indices: [m,3], target_indices: [n,3]
        # compute squared_dist[i,j] = ||pred_indices[i] - target_indices[j]||^2
        # = pred_squared[i] + target_squared[j] - 2*pred_indices[i]·target_indices[j]
        squared_dist = pred_squared + target_squared.t() - 2 * torch.mm(pred_indices, target_indices.t())  # [m,n]
        
        # Get minimum distances in both directions
        dist_pred_to_target, _ = torch.min(squared_dist, dim=1)  # [m]
        dist_target_to_pred, _ = torch.min(squared_dist, dim=0)  # [n]
        
        # CD is the sum of means (we use mean of sqrt of squared distances)
        cd = torch.mean(torch.sqrt(dist_pred_to_target)) + torch.mean(torch.sqrt(dist_target_to_pred))
        cd_scores.append(cd.item())
    
    # Return average CD over the batch
    if len(cd_scores) > 0:
        return sum(cd_scores) / len(cd_scores)
    else:
        return float('inf')

def calculate_chamfer_distance(pred, target, threshold=0.5, samples=1000):
    """
    Calculate Chamfer Distance between predicted and target voxels.
    Automatically uses GPU if tensors are on GPU, otherwise falls back to CPU.
    
    Args:
        pred: predicted voxels (torch.Tensor)
        target: ground truth voxels (torch.Tensor)
        threshold: threshold for binarization
        samples: max number of points to sample (for memory efficiency)
    
    Returns:
        cd_score: average chamfer distance over the batch
    """
    # Check if inputs are on GPU and use GPU implementation when possible
    if pred.is_cuda and target.is_cuda:
        return calculate_chamfer_distance_gpu(pred, target, threshold, samples)
    
    # Otherwise fall back to CPU implementation
    # Move to CPU and convert to numpy
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    batch_size = pred.shape[0]
    cd_scores = []
    
    for i in range(batch_size):
        # Convert voxels to point clouds
        pred_points = voxel_to_points(pred_np[i, 0], threshold)
        target_points = voxel_to_points(target_np[i, 0], threshold)
        
        # If either point cloud is empty, skip
        if len(pred_points) == 0 or len(target_points) == 0:
            continue
        
        # Sample points if there are too many (for memory efficiency)
        if len(pred_points) > samples:
            indices = np.random.choice(len(pred_points), samples, replace=False)
            pred_points = pred_points[indices]
            
        if len(target_points) > samples:
            indices = np.random.choice(len(target_points), samples, replace=False)
            target_points = target_points[indices]
        
        # Calculate Chamfer Distance
        cd = chamfer_distance_numpy(pred_points, target_points)
        cd_scores.append(cd)
    
    # Return average CD over the batch (if any valid scores)
    if len(cd_scores) > 0:
        return np.mean(cd_scores)
    else:
        return float('inf')  # Return infinity if no valid comparisons

def read_pickle(path, G, G_solver, D_, D_solver, E_=None, E_solver=None):
    """
    Load models and optimizers from pickle files.
    Returns the most recent iteration number.
    """
    try:
        files = os.listdir(path)
        file_list = [int(file.split('_')[-1].split('.')[0]) for file in files if file.startswith('G_') and file.endswith('.pkl')]
        file_list.sort()
        
        if not file_list:
            print("No pickle files found.")
            return 0
            
        recent_iter = str(file_list[-1])
        print(f"Continuing from iteration {recent_iter}")

        with open(path + "/G_" + recent_iter + ".pkl", "rb") as f:
            G.load_state_dict(torch.load(f))
        with open(path + "/G_optim_" + recent_iter + ".pkl", "rb") as f:
            G_solver.load_state_dict(torch.load(f))
        with open(path + "/D_" + recent_iter + ".pkl", "rb") as f:
            D_.load_state_dict(torch.load(f))
        with open(path + "/D_optim_" + recent_iter + ".pkl", "rb") as f:
            D_solver.load_state_dict(torch.load(f))
        
        if E_ is not None:
            with open(path + "/E_" + recent_iter + ".pkl", "rb") as f:
                E_.load_state_dict(torch.load(f))
            with open(path + "/E_optim_" + recent_iter + ".pkl", "rb") as f:
                E_solver.load_state_dict(torch.load(f))
        
        # Yeni: Epok bilgisini oku
        if os.path.exists(path + "/epoch_info_" + recent_iter + ".pkl"):
            with open(path + "/epoch_info_" + recent_iter + ".pkl", "rb") as f:
                epoch_info = torch.load(f)
                last_epoch = epoch_info.get('epoch', int(recent_iter))
                print(f"Continuing from epoch {last_epoch}")
                return last_epoch
                
        # Geriye uyumluluk için, dosya yoksa iteration'u epok olarak kabul et
        return int(recent_iter)

    except Exception as e:
        print("Pickle dosyaları okunurken hata:", e)
        return 0


def save_new_pickle(path, iteration, G, G_solver, D_, D_solver, E_=None, E_solver=None):
    """
    Save models, optimizers and epoch information to pickle files.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + "/G_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(G.state_dict(), f)
    with open(path + "/G_optim_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(G_solver.state_dict(), f)
    with open(path + "/D_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(D_.state_dict(), f)
    with open(path + "/D_optim_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(D_solver.state_dict(), f)
    
    if E_ is not None:
        with open(path + "/E_" + str(iteration) + ".pkl", "wb") as f:
            torch.save(E_.state_dict(), f)
        with open(path + "/E_optim_" + str(iteration) + ".pkl", "wb") as f:
            torch.save(E_solver.state_dict(), f)
    
    # Yeni: Epok bilgisini kaydet
    with open(path + "/epoch_info_" + str(iteration) + ".pkl", "wb") as f:
        torch.save({'epoch': iteration}, f)