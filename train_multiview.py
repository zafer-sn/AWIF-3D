import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from utils import make_hyparam_string, read_pickle, save_new_pickle, generateZ, calculate_iou
from utils import calculate_chamfer_distance  # Import Chamfer Distance calculation
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from utils import ShapeNetMultiviewDataset, var_or_cuda, calculate_iou
from model import _G, _D, _E_MultiView
plt.switch_backend("TkAgg")

def KLLoss(z_mu,z_var):
    return (- 0.5 * torch.sum(1 + z_var - torch.pow(z_mu, 2) - torch.exp(z_var)))

def mse_loss(pred, target):
    """Mean Squared Error Loss - Sum over spatial dimensions"""
    return torch.sum(torch.pow((pred - target), 2))

def bce_loss(pred, target):
    """Binary Cross Entropy Loss - Sum over spatial dimensions"""
    # Clamp predictions to avoid numerical instability
    pred = torch.clamp(pred, min=1e-7, max=1-1e-7)
    loss = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
    return torch.sum(loss)

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """Focal Loss - Sum over spatial dimensions, good for sparse voxel structures"""
    # Clamp predictions to avoid numerical instability
    pred = torch.clamp(pred, min=1e-7, max=1-1e-7)
    
    # Calculate BCE loss per voxel
    bce_loss_per_voxel = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
    
    # Calculate focal weight
    p_t = target * pred + (1 - target) * (1 - pred)
    focal_weight = alpha * torch.pow((1 - p_t), gamma)
    
    focal_loss_per_voxel = focal_weight * bce_loss_per_voxel
    return torch.sum(focal_loss_per_voxel)

def dice_loss(pred, target, smooth=1e-6):
    """Dice Loss - Returns loss value (1 - dice_coefficient)"""
    # Flatten the tensors
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # Calculate intersection and union
    intersection = torch.sum(pred_flat * target_flat)
    union = torch.sum(pred_flat) + torch.sum(target_flat)
    
    # Calculate dice coefficient
    dice_coeff = (2.0 * intersection + smooth) / (union + smooth)
    
    # Return dice loss (we want to minimize, so 1 - dice_coeff)
    # Scale by total elements to match magnitude with other losses
    return (1 - dice_coeff) * pred.numel()

def iou_loss(pred, target, smooth=1e-6):
    """IoU Loss - Returns loss value (1 - iou)"""
    # Flatten the tensors
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # Calculate intersection and union
    intersection = torch.sum(pred_flat * target_flat)
    union = torch.sum(pred_flat) + torch.sum(target_flat) - intersection
    
    # Calculate IoU
    iou = (intersection + smooth) / (union + smooth)
    
    # Return IoU loss (we want to minimize, so 1 - iou)
    # Scale by total elements to match magnitude with other losses
    return (1 - iou) * pred.numel()

def awif_loss(pred, target, epoch=1, max_epochs=100, smooth=1e-6):
    """
    Adaptive Weighted IoU-Focal Loss (AWIF) - Bizim yenilikçi loss fonksiyonumuz
    
    Bu loss 4 ana bileşeni kombine eder:
    1. Surface-aware focal component (yüzey voksellerine odaklanma)
    2. IoU-guided weighting (IoU skoruna göre ağırlıklandırma)
    3. Adaptive difficulty progression (eğitim sırasında zorluk artışı)
    4. Boundary enhancement (kenar voksellerini güçlendirme)
    
    Args:
        pred: Predicted voxels
        target: Ground truth voxels
        epoch: Current training epoch
        max_epochs: Total training epochs
        smooth: Smoothing parameter
    """
    batch_size = pred.shape[0]
    total_loss = 0
    
    # Progressive difficulty factor (training sırasında artar)
    difficulty_factor = min(1.0, epoch / (max_epochs * 0.3))  # İlk %30'da yavaş artış
    
    for i in range(batch_size):
        pred_sample = pred[i]
        target_sample = target[i]
        
        # Flatten for calculations
        pred_flat = pred_sample.view(-1)
        target_flat = target_sample.view(-1)
        
        # 1. IoU hesaplama (türevlenebilir soft versiyonu)
        intersection = torch.sum(pred_flat * target_flat)
        union = torch.sum(pred_flat) + torch.sum(target_flat) - intersection
        soft_iou = (intersection + smooth) / (union + smooth)
        
        # 2. Surface voxel detection (6-connectivity kullanarak)
        target_3d = target_sample.squeeze()
        pred_3d = pred_sample.squeeze()
        
        # Padding ekleyerek boundary voksellerini tespit et
        padded_target = torch.nn.functional.pad(target_3d, (1,1,1,1,1,1), mode='constant', value=0)
        
        # 6-neighbor sum ile surface voksellerini bul
        surface_mask = torch.zeros_like(target_3d)
        for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            shifted = padded_target[1+dx:target_3d.shape[0]+1+dx, 
                                  1+dy:target_3d.shape[1]+1+dy, 
                                  1+dz:target_3d.shape[2]+1+dz]
            surface_mask += (target_3d > 0.5) & (shifted < 0.5)
        
        surface_mask = (surface_mask > 0).float().view(-1)
        
        # 3. Adaptive focal component
        # IoU'ya göre alpha değerini ayarla
        adaptive_alpha = 0.25 + (1 - soft_iou) * 0.75  # Düşük IoU = yüksek alpha
        adaptive_gamma = 2.0 + difficulty_factor * 2.0  # Training ilerledikçe artan gamma
        
        # Binary cross entropy base
        pred_clamped = torch.clamp(pred_flat, min=1e-7, max=1-1e-7)
        bce_loss = -(target_flat * torch.log(pred_clamped) + (1 - target_flat) * torch.log(1 - pred_clamped))
        
        # Focal weight calculation
        p_t = target_flat * pred_flat + (1 - target_flat) * (1 - pred_flat)
        focal_weight = adaptive_alpha * torch.pow((1 - p_t), adaptive_gamma)
        focal_component = focal_weight * bce_loss
        
        # 4. Surface enhancement (yüzey voksellerine extra ağırlık)
        surface_weight = 1.0 + surface_mask * 3.0 * difficulty_factor  # Surface vokseller 4x ağırlık
        enhanced_focal = focal_component * surface_weight
        
        # 5. IoU-guided loss balancing
        # Düşük IoU olan örneklere daha fazla ağırlık ver
        sample_weight = 1.0 + (1 - soft_iou) * 2.0
        
        # 6. Dice component (complementary to focal)
        dice_loss_val = 1 - (2.0 * intersection + smooth) / (torch.sum(pred_flat) + torch.sum(target_flat) + smooth)
        
        # 7. Final combination
        # IoU düşükse dice'a daha fazla ağırlık, yüksekse focal'a daha fazla ağırlık
        dice_weight = (1 - soft_iou) * 0.6
        focal_weight_final = soft_iou * 0.4 + 0.6
        
        combined_loss = (
            focal_weight_final * torch.sum(enhanced_focal) + 
            dice_weight * dice_loss_val * pred_flat.numel() +
            (1 - soft_iou) * 100.0  # IoU penalty term
        )
        
        # Sample-specific weighting
        total_loss += combined_loss * sample_weight
    
    return total_loss

def get_reconstruction_loss(loss_type, pred, target, args, epoch=1):
    """Get reconstruction loss based on the specified type"""
    if loss_type == 'mse':
        return mse_loss(pred, target)
    elif loss_type == 'bce':
        return bce_loss(pred, target)
    elif loss_type == 'focal':
        return focal_loss(pred, target, args.focal_alpha, args.focal_gamma)
    elif loss_type == 'dice':
        return dice_loss(pred, target, args.dice_smooth)
    elif loss_type == 'iou':
        return iou_loss(pred, target, args.dice_smooth)
    elif loss_type == 'awif':
        return awif_loss(pred, target, epoch, args.n_epochs, args.dice_smooth)
    else:
        raise ValueError(f"Unknown reconstruction loss type: {loss_type}")

def train_multiview(args):
    hyparam_list = [("model", args.model_name), ("recon_loss", args.recon_loss)]

    hyparam_dict = OrderedDict(((arg, value) for arg, value in hyparam_list))
    log_param = make_hyparam_string(hyparam_dict)
    print(log_param)

    # for using tensorboard
    if args.use_tensorboard:
        import tensorflow as tf
        summary_writer = tf.summary.create_file_writer(args.output_dir + args.log_dir + log_param)

        def inject_summary(summary_writer, tag, value, step):
            with summary_writer.as_default():
                tf.summary.scalar(tag, value, step=step)

    # datset define
    dsets_path = args.input_dir + args.data_dir + "train/"
    print(dsets_path)
    dsets = ShapeNetMultiviewDataset(dsets_path, args)
    dset_loaders = torch.utils.data.DataLoader(dsets, batch_size=args.batch_size, shuffle=True, num_workers=4, prefetch_factor=2, pin_memory=True, persistent_workers=True, drop_last=True)

    # model define
    D = _D(args)
    G = _G(args)
    E = _E_MultiView(args)

    D_solver = optim.Adam(D.parameters(), lr=args.d_lr, betas=args.beta)
    G_solver = optim.Adam(G.parameters(), lr=args.g_lr, betas=args.beta)
    E_solver = optim.Adam(E.parameters(), lr=args.e_lr, betas=args.beta)
    
    if torch.cuda.is_available():
        print("using cuda")
        D.cuda()
        G.cuda()
        E.cuda()

    criterion = nn.BCELoss()

    pickle_path = args.output_dir + args.pickle_dir + log_param
    
    # Değişiklik: read_pickle'dan başlangıç epok sayısını al
    start_epoch = read_pickle(pickle_path, G, G_solver, D, D_solver, E, E_solver) + 1
    print(f"Eğitime {start_epoch}. epoktan devam ediliyor.")

    print(f"Using reconstruction loss: {args.recon_loss}")
    if args.recon_loss == 'focal':
        print(f"Focal loss parameters - alpha: {args.focal_alpha}, gamma: {args.focal_gamma}")
    elif args.recon_loss in ['dice', 'iou']:
        print(f"Smoothing parameter: {args.dice_smooth}")
    elif args.recon_loss == 'awif':
        print(f"Using our innovative AWIF loss - Adaptive Weighted IoU-Focal Loss")
        print(f"This loss adapts to IoU performance and focuses on surface voxels")

    for epoch in range(start_epoch, args.n_epochs+1):
        epoch_start_time = time.time() # Record epoch start time
        # Initialize epoch accumulators
        epoch_iou = 0
        epoch_chamfer_distance = 0  # Add accumulator for Chamfer Distance
        epoch_d_precision = 0
        epoch_d_recall = 0
        epoch_d_f1 = 0
        epoch_d_real_loss = 0
        epoch_d_fake_loss = 0
        epoch_d_loss = 0
        epoch_g_loss = 0
        epoch_d_acu = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        num_batches = 0
        num_cd_valid_batches = 0  # Counter for valid CD calculations

        for i, (images, model_3d) in enumerate(dset_loaders):

            model_3d = var_or_cuda(model_3d)
            # if model_3d.size()[0] != int(args.batch_size):
            #    # print("batch_size != {} drop last incompatible batch".format(int(args.batch_size)))
            #    continue

            Z = generateZ(args)
            Z_vae, z_mus, z_vars = E(images)
            #Z_vae = E.reparameterize(z_mu, z_var)
            G_vae = G(Z_vae)
            
            real_labels = var_or_cuda(torch.Tensor(args.batch_size).uniform_(0.7, 1.0))
            fake_labels = var_or_cuda(torch.Tensor(args.batch_size).uniform_(0, 0.3))

            # ============= Train the discriminator =============#
            d_real = D(model_3d)
            d_real_loss = criterion(d_real.squeeze(), real_labels)

            fake = G(Z)
            d_fake = D(fake)
            d_fake_loss = criterion(d_fake.squeeze(), fake_labels)

            d_loss = d_real_loss + d_fake_loss

            d_real_acu = torch.ge(d_real.squeeze(), 0.5).float()
            d_fake_acu = torch.le(d_fake.squeeze(), 0.5).float()
            d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu), 0))

            # Calculate Precision, Recall, F1 for Discriminator
            y_true_d = torch.cat((torch.ones_like(real_labels), torch.zeros_like(fake_labels)), 0)
            y_pred_d_prob = torch.cat((d_real.squeeze(), d_fake.squeeze()), 0)
            y_pred_d = (y_pred_d_prob >= 0.5).float()

            # Move tensors to CPU and convert to numpy for sklearn metrics
            y_true_d_np = y_true_d.cpu().numpy()
            y_pred_d_np = y_pred_d.cpu().numpy()

            # Calculate metrics, handle zero division
            precision = precision_score(y_true_d_np, y_pred_d_np, zero_division=0)
            recall = recall_score(y_true_d_np, y_pred_d_np, zero_division=0)
            f1 = f1_score(y_true_d_np, y_pred_d_np, zero_division=0)

            if d_total_acu <= args.d_thresh:
                D.zero_grad()
                d_loss.backward()
                D_solver.step()

            # ============= Train the Encoder =============#
            model_3d = model_3d.view(-1, 1, args.cube_len, args.cube_len, args.cube_len)
            
            # Calculate reconstruction loss using selected loss function
            recon_loss_sum = get_reconstruction_loss(args.recon_loss, G_vae, model_3d, args, epoch)
            
            # For reporting purposes, calculate per-sample average
            if args.recon_loss == 'mse':
                recon_loss_per_sample = torch.mean(torch.pow((G_vae - model_3d), 2), dim=(1, 2, 3, 4))
            elif args.recon_loss == 'bce':
                G_vae_clamped = torch.clamp(G_vae, min=1e-7, max=1-1e-7)
                bce_per_voxel = -(model_3d * torch.log(G_vae_clamped) + (1 - model_3d) * torch.log(1 - G_vae_clamped))
                recon_loss_per_sample = torch.mean(bce_per_voxel, dim=(1, 2, 3, 4))
            elif args.recon_loss == 'focal':
                G_vae_clamped = torch.clamp(G_vae, min=1e-7, max=1-1e-7)
                bce_per_voxel = -(model_3d * torch.log(G_vae_clamped) + (1 - model_3d) * torch.log(1 - G_vae_clamped))
                p_t = model_3d * G_vae + (1 - model_3d) * (1 - G_vae)
                focal_weight = args.focal_alpha * torch.pow((1 - p_t), args.focal_gamma)
                focal_per_voxel = focal_weight * bce_per_voxel
                recon_loss_per_sample = torch.mean(focal_per_voxel, dim=(1, 2, 3, 4))
            elif args.recon_loss == 'dice':
                # For dice, report the dice coefficient itself (not loss)
                G_flat = G_vae.view(args.batch_size, -1)
                target_flat = model_3d.view(args.batch_size, -1)
                intersection = torch.sum(G_flat * target_flat, dim=1)
                union = torch.sum(G_flat, dim=1) + torch.sum(target_flat, dim=1)
                dice_coeff = (2.0 * intersection + args.dice_smooth) / (union + args.dice_smooth)
                recon_loss_per_sample = 1 - dice_coeff  # Report as loss
            elif args.recon_loss == 'iou':
                # For IoU, report the IoU score itself (not loss)
                G_flat = G_vae.view(args.batch_size, -1)
                target_flat = model_3d.view(args.batch_size, -1)
                intersection = torch.sum(G_flat * target_flat, dim=1)
                union = torch.sum(G_flat, dim=1) + torch.sum(target_flat, dim=1) - intersection
                iou_score = (intersection + args.dice_smooth) / (union + args.dice_smooth)
                recon_loss_per_sample = 1 - iou_score  # Report as loss
            elif args.recon_loss == 'awif':
                # For AWIF, calculate simplified per-sample loss for reporting
                G_flat = G_vae.view(args.batch_size, -1)
                target_flat = model_3d.view(args.batch_size, -1)
                intersection = torch.sum(G_flat * target_flat, dim=1)
                union = torch.sum(G_flat, dim=1) + torch.sum(target_flat, dim=1) - intersection
                iou_score = (intersection + args.dice_smooth) / (union + args.dice_smooth)
                # Report as composite score (lower is better)
                recon_loss_per_sample = (1 - iou_score) * 100  # Scale for visibility
            
            batch_recon_loss_avg_per_sample = torch.mean(recon_loss_per_sample)

            kl_loss = 0
            for i in range(args.num_views):
                kl_loss += KLLoss(z_vars[i],z_mus[i])

            # Average KL loss over views and batch size (per sample) for reporting
            batch_kl_loss_avg_per_sample = kl_loss / args.num_views / args.batch_size
            # Use total KL loss (summed over batch implicitly by KLLoss) for E_loss calculation
            total_kl_loss_for_E = kl_loss / args.num_views # Average over views only
            E_loss = recon_loss_sum + total_kl_loss_for_E # Use SUM of recon errors and total KL

            E.zero_grad()
            E_loss.backward()
            E_solver.step()
            # =============== Train the generator ===============#

            # Calculate IoU for VAE reconstruction
            batch_iou = calculate_iou(G_vae, model_3d)
            
            # Calculate Chamfer Distance for VAE reconstruction
            # Tüm batch'ler için CD hesapla (ilk 5 batch'i atlamadan)
            try:
                batch_cd = calculate_chamfer_distance(G_vae, model_3d)
                if batch_cd != float('inf'):  # Only accumulate valid CD scores
                    epoch_chamfer_distance += batch_cd
                    num_cd_valid_batches += 1
                    # Log CD değerlerini daha sık göstermek için
                    # if num_batches % 10 == 0 or num_batches < 10:  # İlk 10 batch ve her 10. batch'te log
                    #   print(f"Batch {num_batches} CD: {batch_cd:.6f}")
            except Exception as e:
                print(f"Batch {num_batches} CD calculation error: {e}")
                # Hata durumunda sessizce devam et

            Z = generateZ(args)

            fake = G(Z)
            d_fake = D(fake)
            g_loss = criterion(d_fake.squeeze(), real_labels)

            Z_vae_detached = Z_vae.detach()
            G_vae_new = G(Z_vae_detached)
            # Use selected reconstruction loss for generator training too
            recon_loss_new_sum = get_reconstruction_loss(args.recon_loss, G_vae_new, model_3d, args, epoch)
            g_loss += recon_loss_new_sum

            D.zero_grad()
            G.zero_grad()
            g_loss.backward()
            G_solver.step()

            # Accumulate metrics
            epoch_d_real_loss += d_real_loss.item()
            epoch_d_fake_loss += d_fake_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_d_acu += d_total_acu.item()
            epoch_recon_loss += batch_recon_loss_avg_per_sample.item() * args.batch_size # Accumulate sum based on per-sample avg for reporting
            epoch_kl_loss += batch_kl_loss_avg_per_sample.item() * args.batch_size # Accumulate sum based on per-sample avg for reporting
            epoch_d_precision += precision
            epoch_d_recall += recall
            epoch_d_f1 += f1
            epoch_iou += batch_iou # IoU is already averaged over the batch in the function
            num_batches += 1

        epoch_end_time = time.time() # Record epoch end time
        epoch_duration = epoch_end_time - epoch_start_time # Calculate duration

        # Calculate epoch averages
        avg_iou = epoch_iou / num_batches
        avg_d_precision = epoch_d_precision / num_batches
        avg_d_recall = epoch_d_recall / num_batches
        avg_d_f1 = epoch_d_f1 / num_batches
        avg_d_real_loss = epoch_d_real_loss / num_batches
        avg_d_fake_loss = epoch_d_fake_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_acu = epoch_d_acu / num_batches
        avg_recon_loss = epoch_recon_loss / (num_batches * args.batch_size) # Average per sample
        avg_kl_loss = epoch_kl_loss / (num_batches * args.batch_size) # Average per sample
        
        # Calculate average Chamfer Distance (if any valid calculations)
        avg_cd = epoch_chamfer_distance / num_cd_valid_batches if num_cd_valid_batches > 0 else float('inf')

        # =============== logging each epoch ===============#
        if args.use_tensorboard:
            log_save_path = args.output_dir + args.log_dir + log_param
            if not os.path.exists(log_save_path):
                os.makedirs(log_save_path)

            info = {
                'loss/loss_D_R': avg_d_real_loss,
                'loss/loss_D_F': avg_d_fake_loss,
                'loss/loss_D': avg_d_loss,
                'loss/loss_G': avg_g_loss,
                'loss/acc_D': avg_d_acu,
                'loss/loss_recon': avg_recon_loss,
                'loss/loss_kl': avg_kl_loss,
                'metric/iou': avg_iou,
                'metric/chamfer_distance': avg_cd,  # Add CD to TensorBoard logs
                'metric/d_precision': avg_d_precision,
                'metric/d_recall': avg_d_recall,
                'metric/d_f1': avg_d_f1,
                'time/epoch_duration_sec': epoch_duration # Log epoch duration
            }

            for tag, value in info.items():
                inject_summary(summary_writer, tag, value, epoch)

            summary_writer.flush()

        # =============== each epoch save model or save image ===============#
        # Extended print statement to include reconstruction loss type and Chamfer Distance
        cd_info = f", CD: {avg_cd:.6f}" if avg_cd != float('inf') else ", CD: N/A"
        print(
            f'Epoch-{epoch}; Loss: {args.recon_loss.upper()}, Time: {epoch_duration:.2f}s, IoU: {avg_iou:.4f}, Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}{cd_info}, D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}, D_acu: {avg_d_acu:.4f}, D_Prec: {avg_d_precision:.4f}, D_Rec: {avg_d_recall:.4f}, D_F1: {avg_d_f1:.4f}, D_lr: {D_solver.state_dict()["param_groups"][0]["lr"]:.4f}')        

        if (epoch) % args.pickle_step == 0:
            pickle_save_path = args.output_dir + args.pickle_dir + log_param
            save_new_pickle(pickle_save_path, epoch, G, G_solver, D, D_solver, E, E_solver)
