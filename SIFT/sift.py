import cv2
import numpy as np
import matplotlib.pyplot as plt

# load grayscale
img1 = cv2.imread("example1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("example2.jpg", cv2.IMREAD_GRAYSCALE)

# create SIFT
sift = cv2.SIFT_create()

# detect and compute
kps1, des1 = sift.detectAndCompute(img1, None)
kps2, des2 = sift.detectAndCompute(img2, None)

# FLANN parameters (fast, good for SIFT)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# matching and Lowe's ratio test
matches = flann.knnMatch(des1, des2, k=2)
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:   # Lowe's ratio (tuneable)
        good.append(m)

print("Good matches:", len(good))

# need at least 4 matches for homography
if len(good) >= 4:
    src_pts = np.float32([kps1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is not None:
        matches_mask = mask.ravel().tolist()
        inliers = np.sum(matches_mask)
        inlier_ratio = inliers / len(good)
        print(f"Homography inliers: {inliers}  ratio: {inlier_ratio:.2f}")

        # visualize inliers (green) and outliers (red)
        draw_params_in = dict(matchColor=(0, 255, 0),
                              singlePointColor=None,
                              matchesMask=matches_mask,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        draw_params_out = dict(matchColor=(0, 0, 255),
                               singlePointColor=None,
                               matchesMask=[0 if m else 1 for m in matches_mask],
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        img_inliers = cv2.drawMatches(img1, kps1, img2, kps2, good, None, **draw_params_in)
        img_outliers = cv2.drawMatches(img1, kps1, img2, kps2, good, None, **draw_params_out)
        vis = cv2.addWeighted(img_inliers, 0.7, img_outliers, 0.7, 0)

        # Display with matplotlib (BGR→RGB)
        plt.figure(figsize=(14, 8))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title(f"Inliers (green): {inliers}, Outliers (red): {len(good) - inliers}")
        plt.axis("off")
        plt.show()

        # Optional reprojection error
        src_in = src_pts[np.array(matches_mask, dtype=bool)].reshape(-1, 2)
        dst_in = dst_pts[np.array(matches_mask, dtype=bool)].reshape(-1, 2)
        src_proj = cv2.perspectiveTransform(src_in.reshape(-1, 1, 2), H).reshape(-1, 2)
        reproj_errs = np.linalg.norm(src_proj - dst_in, axis=1)
        print("Mean reprojection error (pixels):", reproj_errs.mean())
    else:
        print("Homography estimation failed (H is None). Visualizing all good matches.")
        vis = cv2.drawMatches(img1, kps1, img2, kps2, good, None,
                              matchColor=(255, 255, 0),
                              singlePointColor=None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
else:
    print("Not enough matches for homography. Visualizing all good matches.")
    vis = cv2.drawMatches(img1, kps1, img2, kps2, good, None,
                          matchColor=(255, 255, 0),
                          singlePointColor=None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
# --- show visualization ---
if vis is not None:
    plt.figure(figsize=(14, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title("Feature matches (green=inliers, red=outliers, yellow=good matches if no H)")
    plt.axis("off")
    plt.show()