import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ASpanFormer.aspanformer import ASpanFormer 
from config.default import get_cfg_defaults
from utils.misc import lower_config
import demo_utils 

# ---------- Configuration ----------
MODEL_TYPE = "indoor"       # 'indoor' or 'outdoor'
RANSAC_THRESH = 3.0          # in pixels; smaller = stricter geometric constraint
# -----------------------------------

def load_image(path: str):
    """Load a grayscale image and convert to torch tensor [1,1,H,W]."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    img = demo_utils.resize(img,1024)
    tensor = torch.from_numpy(img/255.)[None,None].float()
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

def match_with_aspanformer(img1_tensor, img2_tensor, model_type=MODEL_TYPE):
    """Run ASpanFormer feature matching."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_cfg_defaults()
    config.merge_from_file(f'ASpanFormer/config/aspan/{model_type}/aspan_test.py')
    _config = lower_config(config)
    matcher = ASpanFormer(config=_config['aspan'])
    state_dict = torch.load(f'ASpanFormer/weights/{model_type}.ckpt', map_location='cpu')['state_dict']
    matcher.load_state_dict(state_dict,strict=False)
    matcher.eval().to(device)

    batch = {
    "image0": img1_tensor.to(device),
    "image1": img2_tensor.to(device),
}
    with torch.inference_mode():
        matcher(batch,online_resize=True)

    mkpts0 = batch["mkpts0_f"].cpu().numpy()
    mkpts1 = batch["mkpts1_f"].cpu().numpy()
    confidence = batch["mconf"].cpu().numpy()

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

def draw_aspanformer_matches_with_info(img1, img2, mkpts0, mkpts1, mask=None,
                                 confidence=None, file1="image1", file2="image2",
                                 prediction=None):
    """
    Visualize ASpanFormer matches with:
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

    # --- Concatenate images horizontally ---
    H1, W1 = img1_display.shape[:2]
    vis = cv2.hconcat([img1_display, img2_display])
    H_vis, W_vis = vis.shape[:2]

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
    header = np.full((header_height, W_vis, 3), 230, dtype=np.uint8)  # light gray
    footer = np.full((footer_height, W_vis, 3), 30, dtype=np.uint8)   # dark gray / black

    # --- Add filenames above each image ---
    text_y = int(header_height * 0.75)
    cv2.putText(header, file1, (int(W1 * 0.25) - 60, text_y),
                font, font_scale, (0, 0, 0), thickness)
    cv2.putText(header, file2, (int(W1 + W1 * 0.25) - 60, text_y),
                font, font_scale, (0, 0, 0), thickness)

    # --- Add prediction result below images ---
    if prediction is not None:
        pred_text = f"Prediction - Same Person: {prediction}"
        pred_color = (0, 255, 0) if prediction else (0, 0, 255)
        text_size = cv2.getTextSize(pred_text, font, font_scale, thickness)[0]
        text_x = (W_vis - text_size[0]) // 2
        text_y = int(footer_height * 0.7)
        cv2.putText(footer, pred_text, (text_x, text_y),
                    font, font_scale, pred_color, thickness)

    # --- Combine all parts ---
    vis_with_bars = cv2.vconcat([header, vis, footer])
    return vis_with_bars

def show_image(vis, title="ASpanFormer Matches"):
    """Display visualization with Matplotlib."""
    plt.figure(figsize=(14, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

def predict_identity(stats, reproj_error):
    """
    Placeholder for decision logic:
    - Could combine inlier ratio, mean reprojection error, and mean confidence
      into a similarity or verification score.
    """
    if stats["ratio"] > 0.4 and reproj_error < 5.0:
        return True
    elif stats["ratio"] > 0.25:
        return False # uncertain case
    else:
        return False

# ---------- Main Execution ----------
if __name__ == "__main__":
    img1_path = "data/fingervein-index-l-005-1.bmp"
    img2_path = "data/fingervein-index-l-002-1.bmp"

    img1_tensor, img1_gray = load_image(img1_path)
    img2_tensor, img2_gray = load_image(img2_path)

    mkpts0, mkpts1, confidence = match_with_aspanformer(img1_tensor, img2_tensor)

    H, mask, stats = estimate_homography(mkpts0, mkpts1)

    if H is not None:
        print(f"Homography found: {stats['inliers']} inliers ({stats['ratio']:.2f})")
        reproj_error = compute_reprojection_error(H, mkpts0, mkpts1, mask)
        if reproj_error is not None:
            print(f"Mean reprojection error: {reproj_error:.2f} px")

        prediction = predict_identity(stats, reproj_error)
        print(f"Identity prediction: {prediction}")

        vis = draw_aspanformer_matches_with_info(
            img1_gray, img2_gray, mkpts0, mkpts1, mask=mask,
            confidence=confidence, file1=img1_path, file2=img2_path,
            prediction=prediction
        )
        title = "ASpanFormer Matches (green=inliers, red=outliers)"
    else:
        print("Homography estimation failed or not enough matches.")
        prediction = False
        vis = draw_aspanformer_matches_with_info(
            img1_gray, img2_gray, mkpts0, mkpts1,
            confidence=confidence, file1=img1_path, file2=img2_path, prediction=prediction
        )
        title = "ASpanFormer Matches (cyan=matches, no valid homography)"

    show_image(vis, title)
