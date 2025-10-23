import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import kornia.feature as KF

# ---------- Configuration ----------
MODEL_TYPE = "indoor"       # 'indoor' or 'outdoor'
RANSAC_THRESH = 5.0         #  RANSAC reprojection threshold (in pixels) - lower means stricter inlier/outlier criteria

RESIZE = True            # Whether to resize images
RESIZE_TARGET = (640, 480)  # Target size for resizing (width, height)
KEEP_ASPECT = True       # Whether to keep aspect ratio when resizing

ROOT_DIR = "data"
OUTPUT_ROOT = "LoFTR/results"
# -----------------------------------

def load_image(path: str):
    """Load a grayscale image and convert to torch tensor [1,1,H,W]."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    if RESIZE:
        img = resize_image(img, target_size=RESIZE_TARGET, keep_aspect=KEEP_ASPECT)
        print(f"Resized image shape: {img.shape}, dtype: {img.dtype}")
    tensor = torch.from_numpy(img).float()[None, None] / 255.0
    return tensor, img

def resize_image(img, target_size=(640, 480), keep_aspect=False):
    """
    Resize an image either to a fixed size or while keeping aspect ratio.
    
    Args:
        img (np.ndarray): Input image.
        target_size (tuple): (width, height) if keep_aspect=False.
        keep_aspect (bool): Whether to maintain aspect ratio.
        
    Returns:
        np.ndarray: Resized image.
    """
    if keep_aspect:
        h, w = img.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized
    else:
        return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

def match_with_loftr(img1_tensor, img2_tensor, model_type=MODEL_TYPE):
    """Run LoFTR feature matching using Kornia."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = KF.LoFTR(pretrained=model_type).to(device)
    matcher.eval()

    batch = {"image0": img1_tensor.to(device), "image1": img2_tensor.to(device)}
    with torch.inference_mode():
        output = matcher(batch)

    mkpts0 = output["keypoints0"].cpu().numpy()
    mkpts1 = output["keypoints1"].cpu().numpy()
    confidence = output["confidence"].cpu().numpy()

    print(f"Matched keypoints: {len(mkpts0)}")
    return mkpts0, mkpts1, confidence

def estimate_homography(mkpts0, mkpts1):
    """Estimate homography with RANSAC and compute inlier ratio."""
    if len(mkpts0) < 4:
        return None, None, {"inliers": 0, "ratio": 0.0}

    H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, RANSAC_THRESH)
    if H is None or mask is None:
        return None, None, {"inliers": 0, "ratio": 0.0}

    inliers = int(np.sum(mask))
    ratio = inliers / len(mask)
    return H, mask.ravel().tolist(), {"inliers": inliers, "ratio": ratio}

def compute_reprojection_error(H, mkpts0, mkpts1, mask):
    """Compute mean reprojection error for inlier correspondences."""
    if H is None or mask is None:
        return None

    src_in = mkpts0[np.array(mask, dtype=bool)]
    dst_in = mkpts1[np.array(mask, dtype=bool)]
    if len(src_in) == 0:
        return None

    src_proj = cv2.perspectiveTransform(src_in.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(src_proj - dst_in, axis=1)
    return errors.mean()

def draw_loftr_matches_with_info(img1, img2, mkpts0, mkpts1, mask=None,
                                 confidence=None, file1="image1", file2="image2",
                                 prediction=None, gt=None):
    """
    Visualize LoFTR matches with:
    - File names above each image
    - Matches (green=inlier, red=outlier, cyan=confidence)
    - Prediction result below the images
    """

    # --- Convert grayscale to color ---
    if len(img1.shape) == 2:
        img1_display = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        img1_display = img1.copy()

    if len(img2.shape) == 2:
        img2_display = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    else:
        img2_display = img2.copy()

    # --- Match visualization parameters ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    header_height = 40
    footer_height = 40
    vis_width = max(1000, img1_display.shape[1] + img2_display.shape[1])

    # --- Concatenate images horizontally ---
    H1, W1 = img1_display.shape[:2]
    vis = cv2.hconcat([img1_display, img2_display])
    _, W_imgs = vis.shape[:2]

    # --- Draw matches ---
    if mask is not None:
        for (x1, y1), (x2, y2), inlier in zip(mkpts0, mkpts1, mask):
            color = (0, 255, 0) if inlier else (0, 0, 255)
            cv2.circle(vis, (int(x1), int(y1)), 3, color, -1)
            cv2.circle(vis, (int(x2) + W1, int(y2)), 3, color, -1)
            cv2.line(vis, (int(x1), int(y1)), (int(x2) + W1, int(y2)), color, 1)
    else:
        if confidence is None:
            confidence = np.ones(len(mkpts0))
        for (x1, y1), (x2, y2), conf in zip(mkpts0, mkpts1, confidence):
            c = int(255 * (1 - conf))
            color = (255, 255, c)  # cyan intensity
            cv2.circle(vis, (int(x1), int(y1)), 3, color, -1)
            cv2.circle(vis, (int(x2) + W1, int(y2)), 3, color, -1)
            cv2.line(vis, (int(x1), int(y1)), (int(x2) + W1, int(y2)), color, 1)

    # --- Create header and footer bars ---
    header = np.full((header_height, vis_width, 3), 230, dtype=np.uint8)  # light gray
    footer = np.full((footer_height, vis_width, 3), 30, dtype=np.uint8)   # dark gray / black

    # --- Pad and center vis images ---
    if W_imgs < vis_width:
        pad_w = (vis_width - W_imgs) // 2
        vis = cv2.copyMakeBorder(vis, 0, 0, pad_w, vis_width - W_imgs - pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # --- Add filenames above each image ---
    text_y = int(header_height * 0.75)
    cv2.putText(header, file1, (5, text_y),
                font, font_scale, (0, 0, 0), thickness)
    cv2.putText(header, file2, (int(vis_width * 0.5) + 5, text_y),
                font, font_scale, (0, 0, 0), thickness)

    # --- Add prediction result below images ---
    if prediction is not None:
        pred_text = f"Prediction - Same Person: {prediction}"
        pred_color = (0, 255, 0) if prediction == gt else (0, 0, 255)
        text_size = cv2.getTextSize(pred_text, font, font_scale, thickness)[0]
        text_x = (vis_width - text_size[0]) // 2
        text_y = int(footer_height * 0.7)
        cv2.putText(footer, pred_text, (text_x, text_y),
                    font, font_scale, pred_color, thickness)

    # --- Combine all parts ---
    vis_with_bars = cv2.vconcat([header, vis, footer])
    return vis_with_bars

def show_image(vis, title="LoFTR Matches"):
    """Display visualization with Matplotlib."""
    plt.figure(figsize=(14, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

def save_image(vis, save_path, title="LoFTR Matches"):
    """Save visualization with Matplotlib."""
    plt.figure(figsize=(14, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def predict_identity(stats, reproj_error):
    """
    Placeholder for decision logic:
    - Could combine inlier ratio, mean reprojection error, and mean confidence
      into a similarity or verification score.
    """
    if stats["ratio"] > 0.2 and reproj_error < 5.0:
        return True
    elif stats["ratio"] > 0.15:
        return False # uncertain case
    else:
        return False

# ---------- Main Execution ----------
if __name__ == "__main__":
    for modality in ["face", "iris", "hand", "fingervein"]:
        for gt_type in ["same", "different"]:

            gt = True if gt_type == "same" else False
            base_path = os.path.join(ROOT_DIR, modality, gt_type)

            if not os.path.exists(base_path):
                print(f"Skipping missing folder: {base_path}")
                continue

            # Each subfolder (1–5) contains an image pair
            for subfolder in os.listdir(base_path):
                sub_path = os.path.join(base_path, subfolder)
                if not os.path.isdir(sub_path):
                    continue

                images = [os.path.join(sub_path, f) for f in os.listdir(sub_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]

                if len(images) != 2:
                    print(f"⚠️ Skipping {sub_path}: expected 2 images, found {len(images)}")
                    continue

                file1, file2 = sorted(images)
                file1_name = os.path.basename(file1)
                file2_name = os.path.basename(file2)

                print(f"Processing {file1_name} vs {file2_name} ({modality}, {gt_type})")

                img1_tensor, img1_gray = load_image(file1)
                img2_tensor, img2_gray = load_image(file2)

                mkpts0, mkpts1, confidence = match_with_loftr(img1_tensor, img2_tensor)

                H, mask, stats = estimate_homography(mkpts0, mkpts1)

                if H is not None:
                    print(f"Homography found: {stats['inliers']} inliers ({stats['ratio']:.2f})")
                    reproj_error = compute_reprojection_error(H, mkpts0, mkpts1, mask)
                    if reproj_error is not None:
                        print(f"Mean reprojection error: {reproj_error:.2f} px")

                    prediction = predict_identity(stats, reproj_error)
                    print(f"Identity prediction: {prediction}")

                    title = f"LoFTR {MODEL_TYPE}, Matches: {len(mkpts0)}."
                    title += f"\n Inliers: {stats['inliers']}, Ratio: {stats['ratio']:.2f}, GT Same Person: {gt}"
                else:
                    print("Homography estimation failed or not enough matches.")
                    prediction = False
                    title = f"LoFTR {MODEL_TYPE}, NO VALID HOMOGRAPHY FOUND, Matches: {len(mkpts0)}."
                    title += f"\n Inliers: {stats['inliers']}, Ratio: {stats['ratio']:.2f}, GT Same Person: {gt}"
                
                vis = draw_loftr_matches_with_info(img1_gray, img2_gray, mkpts0, mkpts1, mask,
                        confidence=confidence, file1=file1_name, file2=file2_name, prediction=prediction, gt=gt)
                
                # Save visualization
                save_dir = os.path.join(OUTPUT_ROOT, modality, gt_type)
                save_path = os.path.join(save_dir, f"{file1_name}_vs_{file2_name}.png")
                save_image(vis, save_path, title)

                print(f" Saved result: {save_path}")
