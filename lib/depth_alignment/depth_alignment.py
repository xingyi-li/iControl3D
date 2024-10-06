import torch
from PIL import Image
from lib.utils.utils import visualize_depth_numpy, save_image
import numpy as np
from kornia.filters import gaussian_blur2d, median_blur
from scipy.ndimage import median_filter


def scale_shift_linear(rendered_depth, predicted_depth, mask, sky_mask=None, fuse=True):
    if mask.sum() == 0:
        return predicted_depth, predicted_depth

    import cv2

    def dilate(x, k=3):
        x = torch.nn.functional.conv2d(
            x.float()[None, None, ...],
            torch.ones(1, 1, k, k).to(mask.device),
            padding="same"
        )
        return x.squeeze() > 0
    mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
    if sky_mask is not None:
        sky_mask_np = (sky_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
        mask_np = (np.logical_or(mask_np, sky_mask_np) * 255).astype(np.uint8)
        sky_mask = sky_mask.squeeze().to(mask.device)

    mask_np = cv2.dilate(mask_np, np.ones((3, 3), np.uint8), iterations=5)
    mask_np = cv2.erode(mask_np, np.ones((3, 3), np.uint8), iterations=5)
    mask_close = torch.from_numpy(mask_np).to(mask.device).bool()
    mask_close_dilated = dilate(~mask_close, k=3)
    mask_close_only_dilated = mask_close_dilated * mask

    if sky_mask is not None:
        mask_close_only_dilated = ~sky_mask * mask_close_only_dilated

    if mask_close_only_dilated.sum() == 0:
        return predicted_depth

    mask_dilated = dilate(~mask, k=3)
    mask_only_dilated = mask_dilated * mask

    try:
        rendered_disparity = 1 / rendered_depth[mask_close_only_dilated].unsqueeze(-1)
        predicted_disparity = 1 / predicted_depth[mask_close_only_dilated].unsqueeze(-1)

        X = torch.cat([predicted_disparity, torch.ones_like(predicted_disparity)], dim=1)
        XTX_inv = (X.T @ X).inverse()
        # XTX_inv = (X.T @ X).pinverse()
        XTY = X.T @ rendered_disparity
        AB = XTX_inv @ XTY
    except:
        rendered_disparity = 1 / rendered_depth[mask_only_dilated].unsqueeze(-1)
        predicted_disparity = 1 / predicted_depth[mask_only_dilated].unsqueeze(-1)

        X = torch.cat([predicted_disparity, torch.ones_like(predicted_disparity)], dim=1)
        XTX_inv = (X.T @ X).inverse()
        # XTX_inv = (X.T @ X).pinverse()
        XTY = X.T @ rendered_disparity
        AB = XTX_inv @ XTY

    fixed_disparity = (1 / predicted_depth) * AB[0] + AB[1]
    fixed_depth = 1 / fixed_disparity

    if fuse:
        direct_combined_depth = torch.where(mask, rendered_depth, predicted_depth)
        fused_depth = torch.where(mask, rendered_depth, fixed_depth)

        mask_dilated = dilate(mask, k=15)
        mask_dilated_gaussian = gaussian_blur2d(mask_dilated.float()[None, None], (15, 15), (15, 15))

        import torch.nn as nn
        pad = nn.ReplicationPad2d(padding=(9, 9, 9, 9)).to(fused_depth.device)
        rendered_depth_pad = pad(rendered_depth[None, None])[0, 0]

        mask_4_inpainted = pad(mask[None, None].float())[0, 0].bool()

        rendered_inpaint = cv2.inpaint(rendered_depth_pad.cpu().numpy(), ((~mask_4_inpainted).cpu().numpy() * 255).astype(np.uint8), 3,
                                       cv2.INPAINT_NS)[9:-9, 9:-9]

        rendered_inpaint = torch.from_numpy(rendered_inpaint).to(fused_depth.device)
        mask_dilated_gaussian = mask_dilated_gaussian.squeeze()

        fused_depth_smooth = mask_dilated_gaussian * rendered_inpaint + (1 - mask_dilated_gaussian) * fused_depth

        fused_depth_smooth = torch.where(mask, rendered_depth, fused_depth_smooth)

        fused_depth = fused_depth_smooth
        return fused_depth, direct_combined_depth
    else:
        return fixed_depth, fixed_depth
