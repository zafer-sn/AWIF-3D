from collections import OrderedDict
import torch
from torch import optim
from utils import make_hyparam_string, read_pickle, SavePloat_Voxels, calculate_iou, var_or_cuda, save_comparison_plot
from utils import voxel_to_obj, save_image_copy, calculate_chamfer_distance  # Added Chamfer Distance import
import os
import shutil  # For file copying
from train_multiview import KLLoss, get_reconstruction_loss
import matplotlib.pyplot as plt
from utils import ShapeNetMultiviewDataset
from model import _G, _D, _E_MultiView
import numpy as np
from datetime import datetime  # Import datetime for timestamping

np.random.seed(0)
torch.manual_seed(0)

def test_3DVAEGAN_MULTIVIEW(args):
    # Create timestamp for unique file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    hyparam_list = [("model", args.model_name), ("recon_loss", args.recon_loss)]
    hyparam_dict = OrderedDict(((arg, value) for arg, value in hyparam_list))
    log_param = make_hyparam_string(hyparam_dict)
    
    # datset define
    dsets_path = args.input_dir + args.data_dir + "test/"
    print(dsets_path)
    dsets = ShapeNetMultiviewDataset(dsets_path, args)
    dset_loaders = torch.utils.data.DataLoader(dsets, batch_size=args.batch_size, shuffle=True, num_workers=4, prefetch_factor=2, pin_memory=True, persistent_workers=True, drop_last=True)

    # model define
    D = _D(args)
    E = _E_MultiView(args)
    G = _G(args)

    D_solver = optim.Adam(D.parameters(), lr=args.d_lr, betas=args.beta)
    G_solver = optim.Adam(G.parameters(), lr=args.g_lr, betas=args.beta)
    E_solver = optim.Adam(E.parameters(), lr=args.g_lr, betas=args.beta)

    if torch.cuda.is_available():
        print("using cuda")
        D.cuda()
        G.cuda()
        E.cuda()
        
    # Load model checkpoints
    pickle_path = args.output_dir + args.pickle_dir + log_param
    read_pickle(pickle_path, G, G_solver, D, D_solver, E, E_solver)
    
    # Initialize metrics including Chamfer Distance
    total_recon_loss = 0
    total_iou = 0
    total_kl_loss = 0
    total_cd = 0  # Add Chamfer Distance tracking
    num_samples = 0
    
    # List to store batch results for detailed reporting
    batch_results = []
    
    # Create directory for comparison plots
    comparison_image_path = args.output_dir + args.image_dir + 'comparison_plots/'
    if not os.path.exists(comparison_image_path):
        os.makedirs(comparison_image_path)
    
    # Create directory for results if it doesn't exist
    results_dir = args.output_dir + "/results/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    # Create directory for OBJ files and input images
    obj_export_dir = args.output_dir + "/obj_exports/"
    if not os.path.exists(obj_export_dir):
        os.makedirs(obj_export_dir)

    print(f"Testing with reconstruction loss: {args.recon_loss}")
    
    for i, (images, model_3d) in enumerate(dset_loaders):
        X = var_or_cuda(model_3d)
        # Reshape X to 5D (add channel dimension) to match G_vae and calculate_iou expectation
        X = X.view(-1, 1, args.cube_len, args.cube_len, args.cube_len)

        Z_vae, z_mus, z_vars = E(images)
        G_vae = G(Z_vae)

        # Calculate Reconstruction Loss using the same function as training
        batch_recon_loss_sum = get_reconstruction_loss(args.recon_loss, G_vae, X, args, epoch=1)
        batch_recon_loss = batch_recon_loss_sum / X.numel()  # Normalize by total elements for reporting
        
        # Calculate per-sample reconstruction loss for individual metrics
        if args.recon_loss == 'mse':
            recon_loss_per_sample = torch.mean(torch.pow((G_vae - X), 2), dim=(1, 2, 3, 4))
        elif args.recon_loss == 'bce':
            G_vae_clamped = torch.clamp(G_vae, min=1e-7, max=1-1e-7)
            bce_per_voxel = -(X * torch.log(G_vae_clamped) + (1 - X) * torch.log(1 - G_vae_clamped))
            recon_loss_per_sample = torch.mean(bce_per_voxel, dim=(1, 2, 3, 4))
        elif args.recon_loss == 'focal':
            G_vae_clamped = torch.clamp(G_vae, min=1e-7, max=1-1e-7)
            bce_per_voxel = -(X * torch.log(G_vae_clamped) + (1 - X) * torch.log(1 - G_vae_clamped))
            p_t = X * G_vae + (1 - X) * (1 - G_vae)
            focal_weight = args.focal_alpha * torch.pow((1 - p_t), args.focal_gamma)
            focal_per_voxel = focal_weight * bce_per_voxel
            recon_loss_per_sample = torch.mean(focal_per_voxel, dim=(1, 2, 3, 4))
        elif args.recon_loss == 'dice':
            # For dice, calculate per-sample dice loss
            G_flat = G_vae.view(X.size(0), -1)
            target_flat = X.view(X.size(0), -1)
            intersection = torch.sum(G_flat * target_flat, dim=1)
            union = torch.sum(G_flat, dim=1) + torch.sum(target_flat, dim=1)
            dice_coeff = (2.0 * intersection + args.dice_smooth) / (union + args.dice_smooth)
            recon_loss_per_sample = 1 - dice_coeff  # Report as loss
        elif args.recon_loss == 'iou':
            # For IoU, calculate per-sample IoU loss
            G_flat = G_vae.view(X.size(0), -1)
            target_flat = X.view(X.size(0), -1)
            intersection = torch.sum(G_flat * target_flat, dim=1)
            union = torch.sum(G_flat, dim=1) + torch.sum(target_flat, dim=1) - intersection
            iou_score = (intersection + args.dice_smooth) / (union + args.dice_smooth)
            recon_loss_per_sample = 1 - iou_score  # Report as loss
        elif args.recon_loss == 'awif':
            # For AWIF, calculate simplified per-sample loss for reporting
            G_flat = G_vae.view(X.size(0), -1)
            target_flat = X.view(X.size(0), -1)
            intersection = torch.sum(G_flat * target_flat, dim=1)
            union = torch.sum(G_flat, dim=1) + torch.sum(target_flat, dim=1) - intersection
            iou_score = (intersection + args.dice_smooth) / (union + args.dice_smooth)
            recon_loss_per_sample = (1 - iou_score) * 100  # Scale for visibility
        
        print(f"RECON LOSS ({args.recon_loss.upper()}) ITER: {i} - {batch_recon_loss.item():.6f}")
        total_recon_loss += batch_recon_loss.item() * X.size(0)

        # Calculate KL Loss
        kl_loss = 0
        for view_idx in range(args.num_views):
             kl_loss += KLLoss(z_mus[view_idx], z_vars[view_idx])
        batch_kl_loss = kl_loss / args.num_views / X.size(0) # Average over views and batch size
        print(f"KL LOSS ITER: {i} - {batch_kl_loss.item():.4f}")
        total_kl_loss += batch_kl_loss.item() * X.size(0)
        
        # Calculate IoU
        batch_iou = calculate_iou(G_vae, X)
        total_iou += batch_iou * X.size(0)
        
        # Calculate Chamfer Distance
        try:
            batch_cd = calculate_chamfer_distance(G_vae, X)
            if batch_cd != float('inf'):  # Only add valid CD scores
                total_cd += batch_cd * X.size(0)
                print(f"Chamfer Distance ITER: {i} - {batch_cd:.6f}")
            else:
                print(f"Chamfer Distance ITER: {i} - Invalid (empty point clouds)")
        except Exception as e:
            print(f"Error calculating Chamfer Distance: {e}")
            batch_cd = float('inf')
        
        num_samples += X.size(0)
        print("IoU ITER: ", i, " - ", batch_iou)

        # Store batch results with loss type information
        batch_results.append({
            'batch': i,
            'iou': batch_iou,
            'recon_loss': batch_recon_loss.item(),
            'recon_loss_type': args.recon_loss,
            'kl_loss': batch_kl_loss.item(),
            'cd': batch_cd,
            'samples': X.size(0)
        })

        # Save the comparison plot for the first sample in the batch
        save_comparison_plot(images[0][0], G_vae[0], X[0], comparison_image_path, i)
        
        # For each sample in the batch, save input image, generated model, and true model
        for j in range(min(3, X.size(0))):  # Save first 3 samples from each batch to avoid too many files
            sample_dir = f"{obj_export_dir}sample_{i}_{j}/"
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir)
            
            # 1. Save a copy of the input image
            input_filename = f"input_view.png"
            
            # Düzeltme: images[j][0] yerine images[0][j] kullan
            # DataLoader veri yapısı: [görünüm_listesi_0, görünüm_listesi_1, ...]
            # Her görünüm_listesi_i, batch'teki tüm i. görünümleri içerir
            try:
                # İlk görünümü kullan (0. görünümün j. örneği)
                if isinstance(images, list) and len(images) > 0:
                    save_image_copy(images[0][j], sample_dir, input_filename)
                else:
                    # Farklı bir veri yapısı durumunda alternatif erişim
                    print(f"Warning: Unexpected images structure: {type(images)}")
                    if hasattr(images, 'shape'):
                        print(f"Images shape: {images.shape}")
            except Exception as e:
                print(f"Error saving input image for sample {j}: {e}")
                continue  # Bu örneği atla ve diğerine geç
            
            # 2. Convert and save generated model as OBJ 
            try:
                gen_voxel = G_vae[j].detach().cpu().numpy().squeeze()
                gen_obj_content = voxel_to_obj(gen_voxel, threshold=0.5, scale=1.0)
                with open(f"{sample_dir}generated_model.obj", "w") as f:
                    f.write(gen_obj_content)
                
                # 3. Convert and save true model as OBJ
                true_voxel = X[j].detach().cpu().numpy().squeeze()
                true_obj_content = voxel_to_obj(true_voxel, threshold=0.5, scale=1.0)
                with open(f"{sample_dir}true_model.obj", "w") as f:
                    f.write(true_obj_content)
                
                # 4. Save metrics for this sample
                sample_iou = calculate_iou(G_vae[j:j+1], X[j:j+1])
                try:
                    sample_cd = calculate_chamfer_distance(G_vae[j:j+1], X[j:j+1])
                except:
                    sample_cd = float('inf')
                
                with open(f"{sample_dir}metrics.txt", "w") as f:
                    f.write(f"IoU: {sample_iou:.6f}\n")
                    f.write(f"Reconstruction Loss: {recon_loss_per_sample[j].item():.6f}\n")
                    f.write(f"Chamfer Distance: {sample_cd:.6f}\n")  # Add CD to individual metrics
            except Exception as e:
                print(f"Error processing 3D models for sample {j}: {e}")
                continue  # Bu örneği atla ve diğerine geç

    # Calculate final averages
    avg_recon_loss = total_recon_loss / num_samples
    avg_iou = total_iou / num_samples
    avg_kl_loss = total_kl_loss / num_samples
    avg_cd = total_cd / num_samples if total_cd > 0 else float('inf')  # Calculate average CD
    
    # Print summary
    print(f"\n--- Test Results ---")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average Reconstruction Loss: {avg_recon_loss:.4f}")
    print(f"Average KL Loss: {avg_kl_loss:.4f}")
    print(f"Average Chamfer Distance: {avg_cd:.6f}")  # Add CD to summary
    print(f"OBJ files exported to: {obj_export_dir}")
    
    # Save results to file with loss type information
    results_file = f"{results_dir}test_results_{args.recon_loss}_{timestamp}.txt"
    with open(results_file, 'w') as f:
        f.write(f"3D VAE-GAN Test Results\n")
        f.write(f"======================\n\n")
        f.write(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Reconstruction Loss: {args.recon_loss.upper()}\n")
        if args.recon_loss == 'focal':
            f.write(f"Focal Loss Parameters - alpha: {args.focal_alpha}, gamma: {args.focal_gamma}\n")
        elif args.recon_loss in ['dice', 'iou']:
            f.write(f"Smoothing parameter: {args.dice_smooth}\n")
        f.write(f"Dataset: {args.data_dir}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Num Views: {args.num_views}\n")
        f.write(f"Cube Length: {args.cube_len}\n")
        f.write(f"Z Size: {args.z_size}\n\n")
        f.write(f"OBJ files exported to: {obj_export_dir}\n\n")
        
        f.write(f"--- Summary Statistics ---\n")
        f.write(f"Total Samples: {num_samples}\n")
        f.write(f"Average IoU: {avg_iou:.6f}\n")
        f.write(f"Average Reconstruction Loss: {avg_recon_loss:.6f}\n")
        f.write(f"Average KL Loss: {avg_kl_loss:.6f}\n")
        f.write(f"Average Chamfer Distance: {avg_cd:.6f}\n\n")  # Add CD to file output
        
        f.write(f"--- Batch Details ---\n")
        for i, batch in enumerate(batch_results):
            f.write(f"Batch {batch['batch']}: IoU = {batch['iou']:.6f}, {batch['recon_loss_type'].upper()} Loss = {batch['recon_loss']:.6f}, KL Loss = {batch['kl_loss']:.6f}, CD = {batch['cd']:.6f}, Samples = {batch['samples']}\n")
    
    print(f"\nTest results saved to: {results_file}")
    
    # Also save a summary file with loss type information
    summary_file = f"{results_dir}test_summary.txt"
    with open(summary_file, 'a') as f:  # Append mode to accumulate results
        f.write(f"{timestamp} | {args.model_name} | {args.recon_loss.upper()} | IoU: {avg_iou:.6f} | Recon: {avg_recon_loss:.6f} | KL: {avg_kl_loss:.6f} | CD: {avg_cd:.6f} | Views: {args.num_views}\n")
    
    print(f"Summary added to: {summary_file}")
    print(f"Remember to check the generated OBJ files in: {obj_export_dir}")