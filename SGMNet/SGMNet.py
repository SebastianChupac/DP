import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml

from components import load_component

# ---------- Configuration ----------
#TODO move all to config files, or here
RANSAC_THRESH = 3.0          # in pixels; smaller = stricter geometric constraint
CONFIG_PATH = "SGMNet/configs/sgm_config.yaml"
COLOR = False                  # Whether to load color images, Super Point works on grayscale, root on color
# For now, tunable parameters can be set in the config file above
# Matchers: SGM, SG, NN
# Extractors: root, sp
RESIZE = True            # Whether to resize images
RESIZE_TARGET = (640, 480)  # Target size for resizing (width, height)
KEEP_ASPECT = True       # Whether to keep aspect ratio when resizing

ROOT_DIR = "data"
OUTPUT_ROOT = "SGMNet/results"
# -----------------------------------
# Download weights from https://drive.google.com/file/d/1Ca0WmKSSt2G6P7m8YAOlSAHEFar_TAWb/view


def load_image(path: str):
    """Load grayscale and color image."""
    img_g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_c = cv2.imread(path)
    if RESIZE:
        img_g = resize_image(img_g, target_size=RESIZE_TARGET, keep_aspect=KEEP_ASPECT)
        img_c = resize_image(img_c, target_size=RESIZE_TARGET, keep_aspect=KEEP_ASPECT)
        print(f"Resized image shape: {img_g.shape}, dtype: {img_g.dtype}")
    if img_g is None or img_c is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img_c, img_g

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

def match_with_sgmnet(img1, img2, config_path=CONFIG_PATH):
    """Run SGMNet feature matching."""
    with open(config_path, "r") as f:
        demo_config = yaml.safe_load(f)

    # Initialize extractor and matcher
    extractor = load_component("extractor", demo_config["extractor"]["name"], demo_config["extractor"])
    matcher = load_component("matcher", demo_config["matcher"]["name"], demo_config["matcher"])

    size1, size2 = np.flip(np.asarray(img1.shape[:2])), np.flip(np.asarray(img2.shape[:2]))

    kpt1, desc1 = extractor.run(img1)
    kpt2, desc2 = extractor.run(img2)
    data = {
        "x1": kpt1,
        "x2": kpt2,
        "desc1": desc1,
        "desc2": desc2,
        "size1": size1,
        "size2": size2,
    }
    # Run SGMNet matcher
    corr1, corr2 = matcher.run(data)

    print(f"Matched keypoints: {len(corr1)}")
    return corr1, corr2, kpt1, kpt2

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

def draw_sgmnet_matches_with_info(img1, img2, mkpts0, mkpts1, mask=None,
                                 confidence=None, file1="image1", file2="image2",
                                 prediction=None, gt=None):
    """
    Visualize SGMNet matches with:
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
    _, W1 = img1_display.shape[:2]
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

def show_image(vis, title="SGMNet Matches"):
    """Display visualization with Matplotlib."""
    plt.figure(figsize=(14, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

def save_image(vis, save_path, title="SGMNet Matches"):
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
    if stats["ratio"] > 0.4 and reproj_error < 5.0:
        return True
    elif stats["ratio"] > 0.25:
        return False # uncertain case
    else:
        return False

# ---------- Main Execution ----------
if __name__ == "__main__":

    # Load SGMNet config
    with open(CONFIG_PATH, "r") as f:
        demo_config = yaml.safe_load(f)

    extractor_name = demo_config["extractor"]["name"]
    matcher_name = demo_config["matcher"]["name"]

    # Walk through each biometric type
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

                img1_color, img1_gray = load_image(file1)
                img2_color, img2_gray = load_image(file2)
    

                if COLOR:
                    mkpts0, mkpts1, kpt1, kpt2 = match_with_sgmnet(img1_color, img2_color, config_path=CONFIG_PATH)
                else:
                    mkpts0, mkpts1, kpt1, kpt2 = match_with_sgmnet(img1_gray, img2_gray, config_path=CONFIG_PATH)

                confidence = np.ones(len(mkpts0))

                H, mask, stats = estimate_homography(mkpts0, mkpts1)

                if H is not None:
                    print(f"Homography found: {stats['inliers']} inliers ({stats['ratio']:.2f})")
                    reproj_error = compute_reprojection_error(H, mkpts0, mkpts1, mask)
                    if reproj_error is not None:
                        print(f"Mean reprojection error: {reproj_error:.2f} px")

                    prediction = predict_identity(stats, reproj_error)
                    print(f"Identity prediction: {prediction}")

                    title = f"Extractor:  {extractor_name} Matcher: {matcher_name}. Kpts1: {len(kpt1)}, Kpts2: {len(kpt2)}, Matches: {len(mkpts0)}."
                    title += f"\n Inliers: {stats['inliers']}, Ratio: {stats['ratio']:.2f}, GT Same Person: {gt}"
                else:
                    print("Homography estimation failed or not enough matches.")
                    prediction = False

                    title = f"Extractor:  {extractor_name} Matcher: {matcher_name}. NO WALID HOMOGRAPHY FOUND, Kpts1: {len(kpt1)}, Kpts2: {len(kpt2)}, Matches: {len(mkpts0)}."
                    title += f"\n Inliers: {stats['inliers']}, Ratio: {stats['ratio']:.2f}, GT Same Person: {gt}"

                vis = draw_sgmnet_matches_with_info(img1_gray, img2_gray, mkpts0, mkpts1, mask,
                        confidence=confidence, file1=file1_name, file2=file2_name,
                        prediction=prediction, gt=gt)

                # Save visualization
                save_dir = os.path.join(OUTPUT_ROOT, modality, gt_type)
                save_path = os.path.join(save_dir, f"{file1_name}_vs_{file2_name}.png")
                save_image(vis, save_path, title)

                print(f" Saved result: {save_path}")
