import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# ---------- Configuration ----------
METHOD = "ORB"            # choose: "SIFT" or "ORB"
LOWE_RATIO = 0.75         # Lowe's ratio test threshold - heigher allows more matches into "good matches"
RANSAC_THRESH = 5.0       # RANSAC reprojection threshold (in pixels) - lower means stricter inlier/outlier criteria

# SIFT/FLANN parameters
FLANN_TREES = 5
FLANN_CHECKS = 50

# ORB parameters  
ORB_N_FEATURES = 1000
ORB_MATCHER = "BF"     # choose: "BF" (BruteForce) or "FLANN"
USE_CROSS_CHECK = False    # Whether to use cross-check validation (only for BF)

# ORB/FLANN parameters (for when ORB_MATCHER = "FLANN")
FLANN_LSH_TABLE_NUMBER = 6    # LSH table number (typically 6-20)
FLANN_LSH_KEY_SIZE = 12       # Key size (typically 10-20)
FLANN_LSH_MULTI_PROBE_LEVEL = 1  # Multi-probe level (typically 1-2)

# Prediction thresholds (PLACEHOLDER)
RATIO_THRESHOLD = 0.45       # Minimum inlier ratio to predict "same person"

RESIZE = True            # Whether to resize images
RESIZE_TARGET = (640, 480)  # Target size for resizing (width, height)
KEEP_ASPECT = True       # Whether to keep aspect ratio when resizing
# -----------------------------------

def load_image(path: str):
    """Load an image in grayscale."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    if RESIZE:
        img = resize_image(img, target_size=RESIZE_TARGET, keep_aspect=KEEP_ASPECT)
        print(f"Resized image shape: {img.shape}, dtype: {img.dtype}")
    return img

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


def compute_features(image, method=METHOD):
    """Detect and compute keypoints + descriptors using SIFT or ORB."""
    if method.upper() == "SIFT":
        extractor = cv2.SIFT_create()
    elif method.upper() == "ORB":
        extractor = cv2.ORB_create(nfeatures=ORB_N_FEATURES)
    else:
        raise ValueError(f"Unknown method: {method}")

    keypoints, descriptors = extractor.detectAndCompute(image, None)
    return keypoints, descriptors


def match_features(des1, des2, method=METHOD, ratio=LOWE_RATIO):
    """Match descriptors using the right matcher for SIFT/ORB."""
    if method.upper() == "SIFT":
        # Float descriptors → FLANN with KD-tree
        index_params = dict(algorithm=1, trees=FLANN_TREES)
        search_params = dict(checks=FLANN_CHECKS)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        matches = matcher.knnMatch(des1, des2, k=2)

    elif method.upper() == "ORB":
        if ORB_MATCHER.upper() == "FLANN":
            # ORB with FLANN using LSH index for binary descriptors
            index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                               table_number=FLANN_LSH_TABLE_NUMBER,
                               key_size=FLANN_LSH_KEY_SIZE,
                               multi_probe_level=FLANN_LSH_MULTI_PROBE_LEVEL)
            search_params = dict(checks=FLANN_CHECKS)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            matches = matcher.knnMatch(des1, des2, k=2)
            
        else:  # BruteForce
            if USE_CROSS_CHECK:
                # Cross-check: only consistent matches in both directions
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                # Convert to list of lists for consistency with knnMatch format
                matches = [[m] for m in matches]
            else:
                # Standard BF with ratio test
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test (except for cross-check which is already filtered)
    if USE_CROSS_CHECK and method.upper() == "ORB" and ORB_MATCHER.upper() == "BF":
        good = [m[0] for m in matches]  # Cross-check returns direct matches
    else:
        # Add safety check for matches
        good = []
        for match_pair in matches:
            if len(match_pair) == 2:  # Only proceed if we have 2 matches, ORB+FLANN may return less
                m, n = match_pair
                if m.distance < ratio * n.distance:
                    good.append(m)
    
    return good


def estimate_homography(kps1, kps2, matches):
    """Estimate homography using RANSAC and return (H, mask, stats)."""
    if len(matches) < 4:
        return None, None, {"inliers": 0, "ratio": 0}

    src_pts = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESH) # RANSAC determines number of iterations internally

    if H is None or mask is None:
        return None, None, {"inliers": 0, "ratio": 0}

    inliers = int(np.sum(mask))
    ratio = inliers / len(matches)
    return H, mask.ravel().tolist(), {"inliers": inliers, "ratio": ratio}


def draw_matches_with_info(img1, img2, kps1, kps2, matches, mask=None, 
                          file1="image1", file2="image2", prediction=None, gt=None):
    """Visualize matches with file names and prediction result."""
    # Convert grayscale to BGR for colored text if needed
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
    vis = cv2.hconcat([img1_display, img2_display])
    _, W_imgs = vis.shape[:2]
    
    
    # Draw matches
    if mask is not None and any(mask):
        # Inliers (green)
        params_in = dict(matchColor=(0, 255, 0), matchesMask=mask,
                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # Outliers (red)
        params_out = dict(matchColor=(0, 0, 255),
                          matchesMask=[0 if m else 1 for m in mask],
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        img_in = cv2.drawMatches(img1_display, kps1, img2_display, kps2, matches, None, **params_in)
        img_out = cv2.drawMatches(img1_display, kps1, img2_display, kps2, matches, None, **params_out)
        vis = cv2.addWeighted(img_in, 0.6, img_out, 0.6, 0)
    else:
        # All matches (cyan)
        vis = cv2.drawMatches(img1_display, kps1, img2_display, kps2, matches, None,
                              matchColor=(255, 255, 0),
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
    # --- Create header and footer bars ---
    header = np.full((header_height, vis_width, 3), 230, dtype=np.uint8)  # light gray
    footer = np.full((footer_height, vis_width, 3), 30, dtype=np.uint8)   # dark gray / black

    # --- Pad and center images ---
    if W_imgs < vis_width:
        pad_w = (vis_width - W_imgs) // 2
        vis = cv2.copyMakeBorder(vis, 0, 0, pad_w, vis_width - W_imgs - pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))


    # --- Add two filenames to header ---
    text_y = int(header_height * 0.75)
    cv2.putText(header, file1, (5, text_y),
                font, font_scale, (0, 0, 0), thickness)
    cv2.putText(header, file2, (int(vis_width * 0.5 + 5), text_y),
                font, font_scale, (0, 0, 0), thickness)
    
    # Add prediction result at the bottom
    if prediction is not None:
        pred_text = f"Prediction - Same Person: {prediction}"
        pred_color = (0, 255, 0) if prediction == gt else (0, 0, 255)  # Green for correct, Red for incorrect
        
        # Add background for text
        #cv2.rectangle(vis, (0, h-40), (w, h), (0, 0, 0), -1)
        
        # Add prediction text
        text_size = cv2.getTextSize(pred_text, font, font_scale, thickness)[0]
        text_x = (vis_width - text_size[0]) // 2
        cv2.putText(footer, pred_text, (text_x, footer_height - 10), font, font_scale, pred_color, thickness)

    vis_with_bars = cv2.vconcat([header, vis, footer])
    return vis_with_bars


def compute_reprojection_error(H, src_pts, dst_pts, mask):
    """Compute mean reprojection error for inlier points."""
    if H is None:
        return None
    src_in = src_pts[np.array(mask, dtype=bool)].reshape(-1, 2)
    dst_in = dst_pts[np.array(mask, dtype=bool)].reshape(-1, 2)
    if len(src_in) == 0:
        return None
    src_proj = cv2.perspectiveTransform(src_in.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(src_proj - dst_in, axis=1)
    return errors.mean()

#PLACEHOLDER
def predict_same_person(inlier_ratio, ratio_threshold=0.3):
    """Simple prediction based on inlier ratio."""
    if inlier_ratio is None:
        return False
    
    return inlier_ratio >= ratio_threshold


def show_image(vis, title="Feature Matches"):
    """Display visualization with matplotlib."""
    plt.figure(figsize=(14, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


# ---------- Main ----------
if __name__ == "__main__":
    file1 = "data/face/same/4/004_01_02_041_11_crop_128.png"
    file2 = "data/face/same/4/004_01_02_051_16_crop_128.png"
    gt = False  # Ground truth: same person or not

    img1 = load_image(file1)
    img2 = load_image(file2)

    file1_name = os.path.basename(file1)
    file2_name = os.path.basename(file2)

    kps1, des1 = compute_features(img1)
    kps2, des2 = compute_features(img2)
    print(f"{METHOD}: {len(kps1)} keypoints in img1, {len(kps2)} in img2")

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        print(" Less than 2 keypoints in one of the images. Not enough to match. Skipping.")
        good_matches = []
    else:
        good_matches = match_features(des1, des2)
    print(f"Good matches: {len(good_matches)}")
    if (METHOD.upper() == "ORB"):
        print(f"ORB Matcher: {ORB_MATCHER}, Cross-check: {USE_CROSS_CHECK}")

    H, mask, stats = estimate_homography(kps1, kps2, good_matches)

    if H is not None:
        # Compute prediction based on inlier ratio
        prediction = None
        inlier_ratio = stats['ratio']
        print(f"Inliers: {stats['inliers']}  Ratio: {stats['ratio']:.2f}")

        # Compute reprojection error
        src_pts = np.float32([kps1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) 
        dst_pts = np.float32([kps2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) 
        err = compute_reprojection_error(H, src_pts, dst_pts, mask) 
        if err is not None: 
            print(f"Mean reprojection error (pixels): {err:.2f}")

        # Make prediction PLACEHOLDER
        prediction = predict_same_person(inlier_ratio, RATIO_THRESHOLD)
        print(f"Prediction - Same Person: {prediction}")

        vis = draw_matches_with_info(img1, img2, kps1, kps2, good_matches, mask, 
                                   file1_name, file2_name, prediction, gt)
        title = f"{METHOD} Kpts1: {len(kps1)}, Kpts2: {len(kps2)}, Matches: {len(good_matches)}."
        title += f"\n Inliers: {stats['inliers']}, Ratio: {stats['ratio']:.2f}, GT Same Person: {gt}"
    else:
        print("Homography estimation failed or not enough matches.")
        prediction = False
        vis = draw_matches_with_info(img1, img2, kps1, kps2, good_matches, None, 
                                   file1_name, file2_name, prediction, gt)
        title = f"{METHOD} NO VALID HOMOGRAPHY FOUND Kpts1: {len(kps1)}, Kpts2: {len(kps2)}, Matches: {len(good_matches)}."
        title += f"\n Inliers: {stats['inliers']}, Ratio: {stats['ratio']:.2f}, GT Same Person: {gt}"

    show_image(vis, title)