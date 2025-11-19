import os, sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ASpanFormer.aspanformer import ASpanFormer 
from config.default import get_cfg_defaults
from utils.misc import lower_config
import demo_utils

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import VerificationResult
import Utils

# ---------- Configuration ----------
MODEL_TYPE = "indoor"       # 'indoor' or 'outdoor'
RANSAC_THRESH = 5.0         #  RANSAC reprojection threshold (in pixels) - lower means stricter inlier/outlier criteria

#TODO
RESIZE = True            # Whether to resize images
RESIZE_TARGET = (640, 480)  # Target size for resizing (width, height)
KEEP_ASPECT = True       # Whether to keep aspect ratio when resizing

ROOT_DIR = "data"
OUTPUT_ROOT = "ASpanFormer/results"
# -----------------------------------

def load_image(path: str):
    """Load a grayscale image and convert to torch tensor [1,1,H,W]."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    #TODO resize, for now we use demo_utils
    img_resized = demo_utils.resize(img, 512)
    print(f"Resized image shape: {img_resized.shape}, dtype: {img_resized.dtype}")
    return img, img_resized

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
                                 prediction=None, gt=None):
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

def show_image(vis, title="ASpanFormer Matches"):
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
    if stats["ratio"] > 0.4 and reproj_error < 5.0:
        return True
    elif stats["ratio"] > 0.25:
        return False # uncertain case
    else:
        return False

# ---------- Main Execution ----------
if __name__ == "__main__":
    #for modality in ["face", "iris", "hand", "fingervein"]:
    for modality in ["face"]:
        for gt_type in ["same", "different"]:
        #for gt_type in ["same"]:

            gt = True if gt_type == "same" else False
            base_path = os.path.join(ROOT_DIR, modality, gt_type)

            if not os.path.exists(base_path):
                print(f"Skipping missing folder: {base_path}")
                continue

            # Each subfolder (1–5) contains an image pair
            for subfolder in os.listdir(base_path):
            #for subfolder in ["1"]:
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

                img1, img1_resized = load_image(file1)
                img2, img2_resized = load_image(file2)

                image1_mask = None
                image2_mask = None

                if modality == "iris":
                    # Create iris masks
                    image1_mask = Utils.create_iris_mask(img1_resized, exclude_pupil=True)
                    image2_mask = Utils.create_iris_mask(img2_resized, exclude_pupil=True)
                    # Apply masks
                    img1_resized = cv2.bitwise_and(img1_resized, img1_resized, mask=image1_mask)
                    img2_resized = cv2.bitwise_and(img2_resized, img2_resized, mask=image2_mask)
                elif modality == "hand":
                    # Load color images for hand mask creation
                    img1_color = cv2.imread(file1)
                    img2_color = cv2.imread(file2)
                    # resize color images to match resized grayscale images
                    img1_color_resized = demo_utils.resize(img1_color, 512)
                    img2_color_resized = demo_utils.resize(img2_color, 512)
                    # Create hand masks
                    image1_mask = Utils.create_hand_mask(img1_color_resized)
                    image2_mask = Utils.create_hand_mask(img2_color_resized)
                    # Apply masks
                    img1_resized = cv2.bitwise_and(img1_resized, img1_resized, mask=image1_mask)
                    img2_resized = cv2.bitwise_and(img2_resized, img2_resized, mask=image2_mask)
                elif modality == "face":
                    # Load color images for face mask creation
                    img1_color = cv2.imread(file1)
                    img2_color = cv2.imread(file2)
                    img1_color_resized = demo_utils.resize(img1_color, 512)
                    img2_color_resized = demo_utils.resize(img2_color, 512)
                    # Create face masks
                    image1_mask = Utils.create_face_mask(img1_color_resized)
                    image2_mask = Utils.create_face_mask(img2_color_resized)
                    # Apply masks
                    img1_resized = cv2.bitwise_and(img1_resized, img1_resized, mask=image1_mask)
                    img2_resized = cv2.bitwise_and(img2_resized, img2_resized, mask=image2_mask)

                img1_tensor = torch.from_numpy(img1_resized/255.)[None,None].float()
                img2_tensor = torch.from_numpy(img2_resized/255.)[None,None].float()

                mkpts0, mkpts1, confidence = match_with_aspanformer(img1_tensor, img2_tensor)

                # filter matches by confidence threshold - reduces background matches in masked images
                conf_threshold = 0.3
                filtered_mkpts0 = np.array([pt for pt, conf in zip(mkpts0, confidence) if conf >= conf_threshold])
                filtered_mkpts1 = np.array([pt for pt, conf in zip(mkpts1, confidence) if conf >= conf_threshold])
                filtered_confidence = np.array([conf for conf in confidence if conf >= conf_threshold])

                mkpts0 = filtered_mkpts0
                mkpts1 = filtered_mkpts1
                confidence = filtered_confidence

                H, mask, stats = estimate_homography(mkpts0, mkpts1)

                if H is not None:
                    inlier_ratio = stats['ratio']
                    print(f"Homography found: {stats['inliers']} inliers ({stats['ratio']:.2f})")
                    reproj_error = compute_reprojection_error(H, mkpts0, mkpts1, mask)
                    if reproj_error is not None:
                        print(f"Mean reprojection error: {reproj_error:.2f} px")

                    prediction = predict_identity(stats, reproj_error)
                    print(f"Identity prediction: {prediction}")

                    title = f"ASpanFormer {MODEL_TYPE}, Matches: {len(mkpts0)}."
                    title += f"\n Inliers: {stats['inliers']}, Ratio: {stats['ratio']:.2f}, GT Same Person: {gt}"
                else:
                    print("Homography estimation failed or not enough matches.")
                    prediction = False

                    title = f"ASpanFormer {MODEL_TYPE}, NO VALID HOMOGRAPHY FOUND, Matches: {len(mkpts0)}."
                    title += f"\n Inliers: {stats['inliers']}, Ratio: {stats['ratio']:.2f}, GT Same Person: {gt}"

                vis = draw_aspanformer_matches_with_info(img1_resized, img2_resized, mkpts0, mkpts1, mask,
                        confidence=confidence, file1=file1_name, file2=file2_name,
                        prediction=prediction, gt=gt)
                # Save visualization
                save_dir = os.path.join(OUTPUT_ROOT, modality, gt_type)
                save_path = os.path.join(save_dir, f"{file1_name}_vs_{file2_name}.png")
                save_image(vis, save_path, title)

                print(f" Saved result: {save_path}")

                result = VerificationResult.VerificationResult(
                    method_name=f"ASpanFormer_{MODEL_TYPE}",
                    modality=modality,
                    image1=VerificationResult.ImageData(
                        filename=file1_name, 
                        original=img1, 
                        processed=img1_resized,
                        image_type=VerificationResult.ImageType.GRAYSCALE,
                        mask=image1_mask),
                    image2=VerificationResult.ImageData(
                        filename=file2_name,
                        original=img2,
                        processed=img2_resized,
                        image_type=VerificationResult.ImageType.GRAYSCALE,
                        mask=image2_mask),
                    keypoints1= [] if (mkpts0 is None or (np.size(mkpts0) == 0)) else
                                [VerificationResult.Keypoint(x=kp[0], y=kp[1], confidence=None,
                                                           descriptor=None)
                                for kp in mkpts0],
                    keypoints2= [] if (mkpts1 is None or (np.size(mkpts1) == 0)) else
                                [VerificationResult.Keypoint(x=kp[0], y=kp[1], confidence=None,
                                                           descriptor=None)
                                for kp in mkpts1],
                    matches=[VerificationResult.Match(kp1_idx=i,
                                                    kp2_idx=i,
                                                    distance=0.0,  # does not provide distance
                                                    confidence=confidence[i] if confidence is not None else None,
                                                    is_inlier=bool(mask[i]) if mask is not None else None)
                                for i, m in enumerate(mkpts0)],
                    homography=H,
                    homography_confidence=inlier_ratio if H is not None else 0.0,#placeholder
                    inlier_mask=np.array(mask) if mask is not None else None,
                    is_same_person_pred=prediction,
                    verification_confidence=inlier_ratio if H is not None else 0.0,#placeholder
                    ground_truth=gt,
                    num_matches=len(mkpts0),
                    num_inliers=stats['inliers'] if H is not None else 0,
                    inlier_ratio=inlier_ratio if H is not None else 0.0,
                    reprojection_error=reproj_error if H is not None else None
                )

                print(" VerificationResult object created.\n")
                print(result)
