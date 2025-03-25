import torch
import numpy as np

class SimulateHypometabolic(torch.nn.Module):
    def __init__(
        self,
        caps_dir: str,
        pathology: str,
        percentage: int,
        sigma: int = 2    
    ):
        import nibabel as nib

        super(SimulateHypometabolic, self).__init__()

        self.pathology = pathology
        self.percentage = percentage
        self.sigma = sigma

        mask_path = caps_dir / "masks" / f"mask_hypo_{self.pathology.lower()}_resampled.nii"
        mask_nii = nib.load(mask_path)
        self.mask = self.mask_processing(
            mask_nii.get_fdata()
        )

    def forward(self, img):
        new_img = img * self.mask
        return new_img

    def mask_processing(self, mask):
        from scipy.ndimage import gaussian_filter
        inverse_mask = 1 - mask
        inverse_mask[inverse_mask == 0] = 1 - self.percentage / 100
        gaussian_mask = gaussian_filter(inverse_mask, sigma=self.sigma)
        return np.float32(gaussian_mask)