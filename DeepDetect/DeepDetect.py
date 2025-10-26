import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import kornia.feature as KF
import model
from model import ESPNet, CBR, ESPNet_Encoder, InputProjectionA, BR, DownSamplerB, DilatedParllelResidualBlockB, C, CDilated, CB
from PIL import Image
import torchvision.transforms as T


# ---------- Configuration ----------
#MODEL_TYPE = "indoor"       # 'indoor' or 'outdoor'
RANSAC_THRESH = 7        #  RANSAC reprojection threshold (in pixels) - lower means stricter inlier/outlier criteria
MODEL_THRESH = 0.5       # Model's Prediction Threshold (Tau)
NNDR_THRESH = 0.8   # Nearest Neighbour Distance Ratio (NNDR) threshold

# no resing for now as we want to always resize to 480x480
#RESIZE = True            # Whether to resize images
RESIZE_TARGET = (320, 320)  # Target size for resizing (width, height)
# original DeepDetect demo resizes to 480x480
#KEEP_ASPECT = False       # Whether to keep aspect ratio when resizing

ROOT_DIR = "data"
OUTPUT_ROOT = "DeepDetect/results"
# -----------------------------------

def load_image(path: str):
    """Load image and return both original and resized versions."""
    orig_img = cv2.imread(path)
    if orig_img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    
    # Resize for model input (keep original for SIFT computation)
    img_resized = cv2.resize(orig_img, RESIZE_TARGET, interpolation=cv2.INTER_CUBIC)
    print(f"Original image shape: {orig_img.shape}, Resized: {img_resized.shape}")
    
    return orig_img, img_resized



def match_with_deepdetect(orig_img1, orig_img2, img1, img2):
    """Run DeepDetect feature detection and matching - FOLLOWING ORIGINAL SCRIPT APPROACH"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DD_model = model.ESPNet().to(device)
    DD_model = torch.load("DeepDetect/DEEP_DETECT_Best_Model.pth", weights_only=False, map_location=device)  # Load DEEP_DETECT Best Model
    DD_model.eval()


    height_1, width_1 = orig_img1.shape[:2]
    height_2, width_2 = orig_img2.shape[:2]

    img_1 = Image.fromarray(img1)
    img_2 = Image.fromarray(img2)

    transform = T.Compose([T.ToTensor()])
    input_tensor_1 = transform(img_1).unsqueeze(0).to(device)  # [1, 3, 480, 480]
    input_tensor_2 = transform(img_2).unsqueeze(0).to(device)  # [1, 3, 480, 480]

    with torch.no_grad():
        pred_1, pred_2 = DD_model(input_tensor_1), DD_model(input_tensor_2)  # [1, 1, 480, 480]
        pred_1, pred_2 = torch.sigmoid(pred_1), torch.sigmoid(pred_2)  # Converting Logits to Probabilities

    threshold = MODEL_THRESH

    mask_1 = pred_1.cpu().squeeze().numpy()
    mask_2 = pred_2.cpu().squeeze().numpy()
    mask_1 = cv2.resize(mask_1, (width_1, height_1), interpolation=cv2.INTER_CUBIC)
    mask_2 = cv2.resize(mask_2, (width_2, height_2), interpolation=cv2.INTER_CUBIC)
    mask_1 = (mask_1 > threshold).astype(np.uint8)
    mask_2 = (mask_2 > threshold).astype(np.uint8)

    def mask_to_keypoints(mask, size=3):
        ys, xs = np.where(mask == 1)  # get coordinates of ones
        keypoints = [cv2.KeyPoint(float(x), float(y), size) for (y, x) in zip(ys, xs)]
        return keypoints

    kp1_list = mask_to_keypoints(mask_1)    # Creating keypoints from masks
    kp2_list = mask_to_keypoints(mask_2)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.compute(orig_img1, kp1_list)    # Computing SIFT Descriptors for Detected Keypoints
    kp2, des2 = sift.compute(orig_img2, kp2_list)    # Computing SIFT Descriptors for Detected Keypoints

    print(f"Image 1: {len(kp1)} keypoints, Image 2: {len(kp2)} keypoints")
    # img1_kp = cv2.drawKeypoints(orig_img1, kp1, None, (0,242,255), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    # img2_kp = cv2.drawKeypoints(orig_img2, kp2, None, (0,242,255), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    # cv2.imwrite("image_1_DEEPDETECT.png", img1_kp)
    # cv2.imwrite("image_2_DEEPDETECT.png", img2_kp)

    # img1_kp_rgb = cv2.cvtColor(img1_kp, cv2.COLOR_BGR2RGB)   # Convert BGR (OpenCV default) to RGB for displaying correctly with matplotlib
    # img2_kp_rgb = cv2.cvtColor(img2_kp, cv2.COLOR_BGR2RGB)

    # plt.figure(figsize=(8, 4))
    # plt.subplot(1, 2, 1)
    # plt.imshow(img1_kp_rgb)
    # plt.title("Image 1 with DeepDetect Keypoints")
    # plt.axis('off')
    # plt.subplot(1, 2, 2)
    # plt.imshow(img2_kp_rgb)
    # plt.title("Image 2 with DeepDetect Keypoints")
    # plt.axis('off')
    # plt.show()

    if len(kp1) == 0 or len(kp2) == 0:
        return np.array([]), np.array([]), np.array([]), kp1, kp2

    # FLANN Matcher (KD-Tree for SIFT Descriptors)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Nearest Neighbour Distance Ratio (NNDR) Test
    good_matches = []
    nndr_values = []

    eps = 1e-12
    for m, n in matches:
        ratio = m.distance / (n.distance + eps) # Adding eps to avoid division by zero
        nndr_values.append(ratio)
        if ratio < NNDR_THRESH:
            good_matches.append(m)

    print(f"Total good matches after NNDR: {len(good_matches)}")
    
    return good_matches, kp1, kp2, nndr_values

def estimate_homography(mkpts0, mkpts1):
    """Estimate homography with RANSAC and compute inlier ratio."""
    if len(mkpts0) < 4:
        return None, None, {"inliers": 0, "ratio": 0.0}

    H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, RANSAC_THRESH, maxIters=50000)
    if H is None or mask is None:
        return None, None, {"inliers": 0, "ratio": 0.0}

    inliers = int(np.sum(mask))
    ratio = inliers / len(mask)
    return H, mask.ravel().astype(bool).tolist(), {"inliers": inliers, "ratio": ratio}


def compute_reprojection_error(H, mkpts0, mkpts1, mask):
    """Compute mean reprojection error for inlier correspondences."""
    if H is None or mask is None:
        return None
    
    if len(mkpts0) == 0:
        return None

    src_in = mkpts0[np.array(mask, dtype=bool)]
    dst_in = mkpts1[np.array(mask, dtype=bool)]
    if len(src_in) == 0:
        return None

    src_proj = cv2.perspectiveTransform(src_in.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(src_proj - dst_in, axis=1)
    return errors.mean()

def draw_deepdetect_matches_with_info(img1, img2, mkpts0, mkpts1, mask=None,
                                 confidence=None, file1="image1", file2="image2",
                                 prediction=None, gt=None):
    """
    Visualize DeepDetect matches with:
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

    # --- Resize small images for better visualization ---
    target_height = 500
    scale1 = target_height / img1_display.shape[0]
    scale2 = target_height / img2_display.shape[0]
    img1_display = cv2.resize(img1_display, None, fx=scale1, fy=scale1, interpolation=cv2.INTER_LINEAR)
    img2_display = cv2.resize(img2_display, None, fx=scale2, fy=scale2, interpolation=cv2.INTER_LINEAR)

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

    mkpts0 = mkpts0.reshape(-1, 2)
    mkpts1 = mkpts1.reshape(-1, 2)

    # --- Draw matches ---
    kp1 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in mkpts0]
    kp2 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in mkpts1]

    # --- Create cv2.DMatch objects ---
    matches = [cv2.DMatch(i, i, 0) for i in range(len(kp1))]

    # --- Scale coordinates if images were resized ---
    if scale1 != 1.0 or scale2 != 1.0:
        for k in kp1:
            k.pt = (k.pt[0] * scale1, k.pt[1] * scale1)
        for k in kp2:
            k.pt = (k.pt[0] * scale2, k.pt[1] * scale2)

    

    if mask is None:
        mask = [False] * len(matches)

    
    inlier_matches = [m for i, m in enumerate(matches) if mask[i]]
    outlier_matches = [m for i, m in enumerate(matches) if not mask[i]]

    # --- Draw matches ---
    vis_all = cv2.drawMatches(img1_display, kp1, img2_display, kp2, outlier_matches, None,
                              matchColor=(0, 0, 255), singlePointColor=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    vis_inliers = cv2.drawMatches(img1_display, kp1, img2_display, kp2, inlier_matches, None,
                                  matchColor=(0, 255, 0), singlePointColor=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    vis = cv2.addWeighted(vis_all, 0.5, vis_inliers, 0.5, 0)
    vis_width = max(vis.shape[1], vis_width)


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

def show_image(vis, title="DeepDetect Matches"):
    """Display visualization with Matplotlib."""
    plt.figure(figsize=(14, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

def save_image(vis, save_path, title="DeepDetect Matches"):
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
    if stats["ratio"] > 0.3:
        return True
    elif stats["ratio"] > 0.2:
        return False # uncertain case
    else:
        return False



# # ---------- Main Execution ----------
# if __name__ == "__main__":
#     img1_path = 'data/004_01_02_041_11_crop_128.png'
#     img2_path = 'data/004_01_02_051_16_crop_128.png'
#     gt = True  # Ground truth: whether images are of the same person

#     file1_name = os.path.basename(img1_path)
#     file2_name = os.path.basename(img2_path)

#     # Load both original and resized images
#     orig_img1, img1 = load_image(img1_path)
#     orig_img2, img2 = load_image(img2_path)

#     good_matches, kp1, kp2, nndr_values = match_with_deepdetect(orig_img1, orig_img2, img1, img2)
#     matched_img = cv2.drawMatches(orig_img1, kp1, orig_img2, kp2, good_matches, None, matchColor=(0, 255, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#     pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
#     pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

#     #H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=1.0, maxIters=50000)
#     #F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=1.0)  # Use Fundamental Matrix (Epipolar Geometry) to remove outliers - Works best for wide baseline or non-planar scenes
    
#     H, mask, stats = estimate_homography(pts1, pts2)
    
#     inlier_matches = [m for i, m in enumerate(good_matches) if mask[i]]

#     if H is not None:
#         print(f"Homography found: {stats['inliers']} inliers ({stats['ratio']:.2f})")
#         reproj_error = compute_reprojection_error(H, pts1, pts2, mask)
#         if reproj_error is not None:
#             print(f"Mean reprojection error: {reproj_error:.2f} px")

#         prediction = predict_identity(stats, reproj_error)
#         print(f"Identity prediction: {prediction}")

#         title = f"DeepDetect , Matches: {len(good_matches)}."
#         title += f"\n Inliers: {stats['inliers']}, Ratio: {stats['ratio']:.2f}, GT Same Person: {gt}"
#     else:
#         print("Homography estimation failed or not enough matches.")
#         prediction = False
#         title = f"DeepDetect , NO VALID HOMOGRAPHY FOUND, Matches: {len(good_matches)}."
#         title += f"\n Inliers: {stats['inliers']}, Ratio: {stats['ratio']:.2f}, GT Same Person: {gt}"
    
#     vis = draw_deepdetect_matches_with_info(orig_img1, orig_img2, pts1, pts2, mask,
#             confidence=None, file1=file1_name, file2=file2_name, prediction=prediction, gt=gt)
#     show_image(vis, title=title)



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

                # Load both original and resized images
                orig_img1, img1 = load_image(file1)
                orig_img2, img2 = load_image(file2)

                good_matches, kp1, kp2, nndr_values = match_with_deepdetect(orig_img1, orig_img2, img1, img2)

                pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
                pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

                H, mask, stats = estimate_homography(pts1, pts2)

                if H is not None:
                    print(f"Homography found: {stats['inliers']} inliers ({stats['ratio']:.2f})")
                    reproj_error = compute_reprojection_error(H, pts1, pts2, mask)
                    if reproj_error is not None:
                        print(f"Mean reprojection error: {reproj_error:.2f} px")

                    prediction = predict_identity(stats, reproj_error)
                    print(f"Identity prediction: {prediction}")

                    title = f"DeepDetect , Matches: {len(good_matches)}."
                    title += f"\n Inliers: {stats['inliers']}, Ratio: {stats['ratio']:.2f}, GT Same Person: {gt}"
                else:
                    print("Homography estimation failed or not enough matches.")
                    prediction = False
                    title = f"DeepDetect , NO VALID HOMOGRAPHY FOUND, Matches: {len(good_matches)}."
                    title += f"\n Inliers: {stats['inliers']}, Ratio: {stats['ratio']:.2f}, GT Same Person: {gt}"
                
                vis = draw_deepdetect_matches_with_info(orig_img1, orig_img2, pts1, pts2, mask,
                    confidence=None, file1=file1_name, file2=file2_name, prediction=prediction, gt=gt)
                
                # Save visualization
                save_dir = os.path.join(OUTPUT_ROOT, modality, gt_type)
                save_path = os.path.join(save_dir, f"{file1_name}_vs_{file2_name}.png")
                save_image(vis, save_path, title)

                print(f" Saved result: {save_path}")
            

    



