import torch
import torch.nn as nn    
import torch.nn.functional as F

class SimpleMaskedMAE(nn.Module):
    def __init__(self, threshold_pct=0.1):
        super().__init__()
        self.threshold = threshold_pct

    def forward(self, pred, target):
        # 1. Find the max dose for each volume in the batch
        # Keep dims so it broadcasts correctly: (B, 1, 1, 1, 1)
        batch_max = target.amax(dim=(2, 3, 4), keepdim=True)
        
        # 2. Create a boolean mask of the high-dose region
        mask = target >= (batch_max * self.threshold)
        
        # 3. Failsafe: If mask is somehow empty, just do a global MAE
        if mask.sum() == 0:
            return F.l1_loss(pred, target)
            
        # 4. Extract only the masked voxels (flattens into a 1D list automatically)
        # and compute the native PyTorch mean absolute error.
        return F.l1_loss(pred[mask], target[mask])


class SimpleIDDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # 1. Collapse the lateral spatial dimensions (X=2, Y=3) using MEAN
        # This reduces (B, 1, 64, 64, 32) -> (B, 1, 32)
        pred_curve = pred.mean(dim=(2, 3))
        target_curve = target.mean(dim=(2, 3))
        
        # 2. Compute the mean absolute error between the two 1D curves
        return F.l1_loss(pred_curve, target_curve)



import torch

def compute_angle_agnostic_idd(dose_volume, ray_source, ray_target, spacing=(4.0, 4.0, 3.0), bin_size_mm=4.0):
    """
    dose_volume: (B, 1, X, Y, Z) - The predicted or ground truth dose
    ray_source: (B, 3) - Physical coordinates of the beam source
    ray_target: (B, 3) - Physical coordinates of the beam target
    spacing: Tuple of the physical voxel dimensions
    bin_size_mm: The resolution of the 1D output curve
    """
    B, _, X, Y, Z = dose_volume.shape
    device = dose_volume.device
    
    # 1. Create the 3D grid of physical coordinates (matching your Resize dimensions)
    # Shape of each: (X, Y, Z)
    grid_x, grid_y, grid_z = torch.meshgrid(
        torch.arange(X, device=device) * spacing[0],
        torch.arange(Y, device=device) * spacing[1],
        torch.arange(Z, device=device) * spacing[2],
        indexing='ij'
    )
    # Stack into a single coordinate tensor: (3, X, Y, Z)
    coords = torch.stack([grid_x, grid_y, grid_z], dim=0)
    # Flatten spatial dimensions for easier batch math: (B, 3, N_voxels)
    coords = coords.view(1, 3, -1).expand(B, -1, -1) 
    
    # Flatten the dose volume: (B, N_voxels)
    dose_flat = dose_volume.view(B, -1)
    
    # 2. Calculate the normalized beam vector
    beam_vec = ray_target - ray_source # (B, 3)
    beam_len = torch.norm(beam_vec, dim=-1, keepdim=True)
    beam_dir = beam_vec / (beam_len + 1e-6) # (B, 3)
    
    # 3. Project voxel coordinates onto the beam line to get depth
    # Reshape for broadcasting: source (B, 3, 1), dir (B, 3, 1)
    source_b = ray_source.unsqueeze(-1)
    dir_b = beam_dir.unsqueeze(-1)
    
    # Depth = dot_product(coords - source, beam_dir)
    # Resulting shape: (B, N_voxels)
    depths = torch.sum((coords - source_b) * dir_b, dim=1)
    
    # 4. Discretize depths into integer bins (e.g., every 2mm)
    # We clamp at 0 to ignore voxels located "behind" the source
    depth_bins = (depths / bin_size_mm).clamp(min=0).long()
    
    # Define the maximum length of the 1D curve (e.g., 400mm / 2mm bins = 200 bins)
    num_bins = 200 
    depth_bins = depth_bins.clamp(max=num_bins - 1)
    
    # 5. Integrate (Sum) the dose laterally using scatter_add
    # This collapses the 3D volume into a 1D curve per batch item
    idd_curves = torch.zeros((B, num_bins), dtype=torch.float32, device=device)
    idd_curves.scatter_add_(dim=1, index=depth_bins, src=dose_flat)
    
    return idd_curves
    

class BraggPeakPositionLoss(nn.Module):
    def __init__(self, temperature=10.0):
        super().__init__()
        # A higher temperature makes the peak isolation sharper.
        # 10.0 is usually a sweet spot for suppressing the dose plateau.
        self.temperature = temperature 

    def forward(self, pred, target):
        """
        pred, target shape: (Batch, 1, X, Y, Z)
        """
        B, _, X, Y, Z = pred.shape
        device = pred.device
        
        # 1. Scale the volumes between 0 and 1
        # We add 1e-6 to prevent division by zero in empty volumes
        pred_max = pred.amax(dim=(2, 3, 4), keepdim=True) + 1e-6
        target_max = target.amax(dim=(2, 3, 4), keepdim=True) + 1e-6
        
        pred_scaled = F.relu(pred) / pred_max
        target_scaled = F.relu(target) / target_max
        
        # 2. Isolate the Bragg Peak using the temperature power
        # The plateau vanishes, leaving only the high-dose peak
        pred_weight = torch.pow(pred_scaled, self.temperature)
        target_weight = torch.pow(target_scaled, self.temperature)
        
        # 3. Normalize into a spatial probability distribution (sums to 1.0)
        pred_prob = pred_weight / pred_weight.sum(dim=(2, 3, 4), keepdim=True)
        target_prob = target_weight / target_weight.sum(dim=(2, 3, 4), keepdim=True)
        
        # 4. Generate Coordinate Grids (Normalized from -1 to 1 for numerical stability)
        grid_x, grid_y, grid_z = torch.meshgrid(
            torch.linspace(-1, 1, X, device=device),
            torch.linspace(-1, 1, Y, device=device),
            torch.linspace(-1, 1, Z, device=device),
            indexing='ij'
        )
        # Reshape to (1, 3, X, Y, Z) to broadcast across the batch
        grid = torch.stack([grid_x, grid_y, grid_z], dim=0).unsqueeze(0)
        
        # 5. Calculate Center of Mass (The Expected 3D Coordinate)
        # Multiply the probability map by the grid and sum the spatial dimensions
        pred_com = (pred_prob.unsqueeze(1) * grid).sum(dim=(3, 4, 5))    # Shape: (B, 3)
        target_com = (target_prob.unsqueeze(1) * grid).sum(dim=(3, 4, 5)) # Shape: (B, 3)
        
        # 6. Calculate the L2 Distance (MSE) between the predicted and true peak positions
        position_loss = F.mse_loss(pred_com, target_com)
        
        return position_loss

class Level1LossFunction(nn.Module):
    def __init__(self,masked_factor=1.0, iid_curve_weight=0.001, allMAE_weight=1.0, use_high_dose_mask=False, high_dose_threshold=0.8, high_dose_weight=1.0,bragg_peak_weight=50.0):
        super().__init__()
        self.beam_masked_mae_loss = SimpleMaskedMAE()
        self.bragg_peak_loss = BraggPeakPositionLoss()
        #self.IID_curve_loss = SimpleIDDLoss()
        self.allMAE = torch.nn.L1Loss()
        if use_high_dose_mask:
            self.beam_masked_mae_loss = SimpleMaskedMAE(threshold_pct=high_dose_threshold)
            self.high_dose_weight = high_dose_weight
        self.masked_factor = masked_factor
        self.iid_curve_weight = iid_curve_weight
        self.allMAE_weight = allMAE_weight
        self.use_high_dose_mask = use_high_dose_mask
        self.bragg_peak_weight = bragg_peak_weight
    
    def __call__(self, pred_dose, gt_dose,ray_source=None, ray_target=None, fine_tune = False):
        beam_masked_mae = self.beam_masked_mae_loss(pred_dose, gt_dose)
        bragg_peak_loss = self.bragg_peak_loss(pred_dose, gt_dose)
        #idd_curve_loss_value = #self.IID_curve_loss(pred_dose, gt_dose)
        allMAE = self.allMAE(pred_dose, gt_dose)
        if ray_source is not None and ray_target is not None:
            pred_idd = compute_angle_agnostic_idd(pred_dose, ray_source, ray_target)
            target_idd = compute_angle_agnostic_idd(gt_dose, ray_source, ray_target)
            idd_mae = F.l1_loss(pred_idd, target_idd)
        #total_variation = self.total_variation_loss(pred_dose, gt_dose)
        eff_masked = beam_masked_mae * self.masked_factor
        eff_idd = idd_mae * self.iid_curve_weight
        eff_all = allMAE * self.allMAE_weight
        eff_bragg = bragg_peak_loss * self.bragg_peak_weight  # You can adjust the weight for Bragg Peak Loss if needed
        total_loss = eff_masked  + eff_all  #+ total_variation + eff_idd
        if fine_tune:
            total_loss = total_loss + eff_bragg
        if self.use_high_dose_mask:
            high_beam_masked_mae = self.beam_masked_mae_loss(pred_dose, gt_dose)
            eff_high_masked = high_beam_masked_mae * self.high_dose_weight
            total_loss = total_loss + eff_high_masked
        
        lossDict = {
            "masked_mae": beam_masked_mae,
            "idd_curve_loss_value": idd_mae,
            "allMAE": allMAE,
            "bragg_peak_loss": bragg_peak_loss,
            "effective_beam_masked_mae": eff_masked,
            "effective_idd_curve_loss": eff_idd,
            "effective_allMAE": eff_all,
            'effective_bragg_peak_loss': eff_bragg,
            "total_loss": total_loss
        }

        if self.use_high_dose_mask:
            lossDict["high_beam_masked_mae"] = high_beam_masked_mae
            lossDict["effective_high_beam_masked_mae"] = eff_high_masked
            
        return lossDict