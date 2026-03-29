import os
import cv2
import numpy as np
import torch
import lpips
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from tqdm import tqdm

def calculate_metrics(gt_path, pred_path, loss_fn_alex):
    """Calculates PSNR, SSIM, and LPIPS between two images."""
    # Read images (BGR)
    gt_img = cv2.imread(str(gt_path))
    pred_img = cv2.imread(str(pred_path))

    if gt_img is None or pred_img is None:
        raise ValueError(f"Could not read images at {gt_path} or {pred_path}")

    # Ensure same size (if models crop or pad slightly, we crop to minimum dimensions)
    h_gt, w_gt = gt_img.shape[:2]
    h_pr, w_pr = pred_img.shape[:2]
    if h_gt != h_pr or w_gt != w_pr:
        h, w = min(h_gt, h_pr), min(w_gt, w_pr)
        gt_img = gt_img[:h, :w, :]
        pred_img = pred_img[:h, :w, :]

    # Calculate PSNR and SSIM on RGB arrays
    gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
    pred_rgb = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)

    psnr_val = compute_psnr(gt_rgb, pred_rgb, data_range=255)
    ssim_val = compute_ssim(gt_rgb, pred_rgb, channel_axis=-1, data_range=255)

    # Calculate LPIPS
    # lpips requires tensors in [-1, 1] range, channel first (N, C, H, W)
    gt_tensor = lpips.im2tensor(gt_rgb)
    pred_tensor = lpips.im2tensor(pred_rgb)

    # Disable gradients for faster/less memory inference
    with torch.no_grad():
        lpips_val = loss_fn_alex(gt_tensor, pred_tensor).item()

    return psnr_val, ssim_val, lpips_val

def main():
    GT_DIR = 'ground_truth'
    OUTPUT_DIR = 'output'
    MODELS = ['4x-ESRGAN', '4x-SPAN', '4x-Transformer'] # prefixes used in output saving

    if not os.path.exists(GT_DIR):
        print(f"⚠️ Directory '{GT_DIR}' not found. Please place your high-res original images here.")
        os.makedirs(GT_DIR, exist_ok=True)
        return

    # Initialize LPIPS model once
    print("Loading LPIPS metric model (AlexNet)...")
    loss_fn_alex = lpips.LPIPS(net='alex', version='0.1')

    # Get Ground Truth files
    formats = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    gt_files = [f for f in Path(GT_DIR).iterdir() if f.suffix.lower() in formats]

    if not gt_files:
         print(f"⚠️ No ground truth images found in '{GT_DIR}'.")
         return
         
    print(f"\nFound {len(gt_files)} Ground Truth images to evaluate against.")

    results = {model: {'psnr': [], 'ssim': [], 'lpips': []} for model in MODELS}

    for gt_path in gt_files:
        # The upscaled images in output are prepended with the model name: e.g., "4x-SPAN_image.jpg"
        for model in MODELS:
            pred_filename = f"{model}_{gt_path.name}"
            pred_path = Path(OUTPUT_DIR) / pred_filename

            if not pred_path.exists():
                # print(f"Warning: Missing output for {model} on image {gt_path.name}")
                continue
            
            try:
                psnr, ssim, lpip = calculate_metrics(gt_path, pred_path, loss_fn_alex)
                results[model]['psnr'].append(psnr)
                results[model]['ssim'].append(ssim)
                results[model]['lpips'].append(lpip)
            except Exception as e:
                print(f"Error comparing {gt_path.name} with {model}: {e}")

    import json
    with open('evaluation_results.json', 'w') as f:
        # compute averages
        final_res = {}
        for m in MODELS:
            v = results[m]
            if v['psnr']:
                final_res[m] = {
                    'PSNR': np.mean(v['psnr']),
                    'SSIM': np.mean(v['ssim']),
                    'LPIPS': np.mean(v['lpips'])
                }
        json.dump(final_res, f, indent=4)


if __name__ == '__main__':
    main()
