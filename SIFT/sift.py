import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ---------- Configuration ----------
LOWE_RATIO = 0.75        # Lowe's ratio test threshold - heigher allows more matches into "good matches"
RANSAC_THRESH = 20.0      # RANSAC reprojection threshold (in pixels) - lower means stricter inlier/outlier criteria
FLANN_TREES = 10
FLANN_CHECKS = 100
# -----------------------------------


def load_image(path: str):
    """Load an image in grayscale."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def compute_sift_features(image):
    """Detect and compute SIFT keypoints and descriptors."""
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def match_features(des1, des2, ratio=LOWE_RATIO):
    """Match SIFT descriptors using FLANN and apply Lowe’s ratio test."""
    index_params = dict(algorithm=1, trees=FLANN_TREES) 
    search_params = dict(checks=FLANN_CHECKS)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2) # k - how many nearest neighbors to find for each descriptor
    print(f"Total matches found: {len(matches)}")
    good_matches = [m for m, n in matches if m.distance < ratio * n.distance]
    return good_matches


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


def draw_matches(img1, img2, kps1, kps2, matches, mask=None):
    """Visualize matches: inliers (green), outliers (red), or all (yellow)."""
    if mask is not None and any(mask):
        # Inliers (green)
        params_in = dict(matchColor=(0, 255, 0), matchesMask=mask,
                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # Outliers (red)
        params_out = dict(matchColor=(0, 0, 255),
                          matchesMask=[0 if m else 1 for m in mask],
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        img_in = cv2.drawMatches(img1, kps1, img2, kps2, matches, None, **params_in)
        img_out = cv2.drawMatches(img1, kps1, img2, kps2, matches, None, **params_out)
        vis = cv2.addWeighted(img_in, 0.6, img_out, 0.6, 0)
    else:
        # All matches (cyan)
        vis = cv2.drawMatches(img1, kps1, img2, kps2, matches, None,
                              matchColor=(255, 255, 0), # cyan in BGR
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return vis


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


def show_image(vis, title="Feature Matches"):
    """Display visualization with matplotlib."""
    plt.figure(figsize=(14, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


# ---------- Main Execution ----------
if __name__ == "__main__":
    Image.open("C:/Users/sebas/Documents/VUT FIT MIT/DP/PublicDataset/PublicDataset/FingerVein/001/__MACOSX/Finger Vein Database/001/left/._index_1.bmp").show()
    img1 = load_image("SIFT/fingervein-index-a-1.bmp")
    img2 = load_image("SIFT/fingervein-index-a-2.bmp")

    kps1, des1 = compute_sift_features(img1)
    kps2, des2 = compute_sift_features(img2)

    good_matches = match_features(des1, des2)
    print(f"Good matches: {len(good_matches)}")

    H, mask, stats = estimate_homography(kps1, kps2, good_matches)

    if H is not None:
        print(f"Homography inliers: {stats['inliers']}  ratio: {stats['ratio']:.2f}")

        # Compute reprojection error
        src_pts = np.float32([kps1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kps2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        err = compute_reprojection_error(H, src_pts, dst_pts, mask)
        if err is not None:
            print(f"Mean reprojection error (pixels): {err:.2f}")

        vis = draw_matches(img1, img2, kps1, kps2, good_matches, mask)
        title = "Feature Matches (green=inliers, red=outliers)"
    else:
        print("Homography estimation failed or not enough matches.")
        vis = draw_matches(img1, img2, kps1, kps2, good_matches)
        title = "Feature Matches (cyan=good matches, no valid homography)"

    show_image(vis, title)
