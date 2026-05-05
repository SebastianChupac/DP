"""Render/save/show utilities for single-pair matcher visualization."""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from ..results import VisualizationResult, Keypoint, Match

DEFAULT_MAX_MATCHES = 5000
MIN_CANVAS_WIDTH = 1100
ANNOTATION_ALPHA = 0.7


def _to_bgr(img: np.ndarray) -> np.ndarray:
    """Ensure image is BGR uint8 for OpenCV drawing."""
    if img is None:
        return np.zeros((64, 64, 3), dtype=np.uint8)

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.ndim == 3 and img.shape[2] == 3:
        return img.copy()

    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return np.zeros((64, 64, 3), dtype=np.uint8)


def _cv2_keypoints(points: List[Keypoint]) -> List[cv2.KeyPoint]:
    """Convert unified keypoints to cv2.KeyPoint list."""
    out: List[cv2.KeyPoint] = []
    for kp in points:
        size = float(kp.size) if kp.size is not None else 4.0
        angle = float(kp.angle) if kp.angle is not None else -1.0
        response = float(kp.response) if kp.response is not None else 0.0
        octave = int(kp.octave) if kp.octave is not None else 0
        class_id = int(kp.class_id) if kp.class_id is not None else -1
        out.append(cv2.KeyPoint(float(kp.x), float(kp.y), size, angle, response, octave, class_id))
    return out


def _truncate_to_pixel_width(text: str, max_width: int, font, scale: float, thickness: int) -> str:
    """Truncate text with ellipsis so it fits in max pixel width."""
    if max_width <= 0:
        return ""

    w = cv2.getTextSize(text, font, scale, thickness)[0][0]
    if w <= max_width:
        return text

    ellipsis = "..."
    low = 0
    high = len(text)
    best = ellipsis

    while low <= high:
        mid = (low + high) // 2
        candidate = text[:mid] + ellipsis
        cw = cv2.getTextSize(candidate, font, scale, thickness)[0][0]
        if cw <= max_width:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1

    return best


def _draw_segmented_text(
    canvas: np.ndarray,
    x: int,
    y: int,
    font,
    scale: float,
    segments: List[Tuple[str, Tuple[int, int, int], int]],
) -> None:
    """Draw text as sequential segments with individual color/thickness."""
    cursor_x = x
    for text, color, thickness in segments:
        cv2.putText(canvas, text, (cursor_x, y), font, scale, color, thickness, cv2.LINE_AA)
        text_w = cv2.getTextSize(text, font, scale, thickness)[0][0]
        cursor_x += text_w


def _selected_matches(result: VisualizationResult) -> Tuple[List[Match], int, int, bool]:
    """Select matches to draw and update cutoff metadata in the result."""
    max_matches = DEFAULT_MAX_MATCHES

    total_matches = len(result.matches)

    # Prefer strongest/closest matches when clipping.
    ordered = sorted(result.matches, key=lambda m: float(m.distance))
    selected = ordered[:max_matches]

    drawn = len(selected)
    cutoff_applied = drawn < total_matches

    result.metadata["visualization_max_matches"] = max_matches
    result.metadata["visualization_total_matches"] = total_matches
    result.metadata["visualization_drawn_matches"] = drawn
    result.metadata["visualization_cutoff_applied"] = cutoff_applied

    return selected, total_matches, drawn, cutoff_applied


def _get_unmatched_keypoints(result: VisualizationResult, selected_matches: List[Match]) -> Tuple[List[int], List[int]]:
    """Get keypoint indices that were not matched in selected_matches.
    
    Returns:
        Tuple of (unmatched_kp1_indices, unmatched_kp2_indices)
    """
    matched_kp1 = set(m.kp1_idx for m in selected_matches if m.kp1_idx >= 0)
    matched_kp2 = set(m.kp2_idx for m in selected_matches if m.kp2_idx >= 0)
    
    unmatched_kp1 = [i for i in range(len(result.keypoints1)) if i not in matched_kp1]
    unmatched_kp2 = [i for i in range(len(result.keypoints2)) if i not in matched_kp2]
    
    return unmatched_kp1, unmatched_kp2


def _draw_keypoints_overlay(
    vis: np.ndarray,
    kpts1: List[cv2.KeyPoint],
    kpts2: List[cv2.KeyPoint],
    left_width: int,
    indices1: List[int],
    indices2: List[int],
    color: Tuple[int, int, int] = (0, 255, 255),
    radius: int = 3,
) -> None:
    """Draw selected keypoints on top of a concatenated image pair."""
    for idx in indices1:
        if 0 <= idx < len(kpts1):
            kp = kpts1[idx]
            cv2.circle(vis, (int(kp.pt[0]), int(kp.pt[1])), radius, color, -1)

    for idx in indices2:
        if 0 <= idx < len(kpts2):
            kp = kpts2[idx]
            cv2.circle(vis, (int(kp.pt[0]) + left_width, int(kp.pt[1])), radius, color, -1)


def _blend_annotations(base: np.ndarray, overlay: np.ndarray, alpha: float = ANNOTATION_ALPHA) -> np.ndarray:
    """Blend annotation overlay back onto the base image."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0.0:
        return base
    if alpha >= 1.0:
        return overlay
    return cv2.addWeighted(base, 1.0 - alpha, overlay, alpha, 0.0)


def _apply_transform_to_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply 3x3 homography-like transform to Nx2 points."""
    if points.size == 0:
        return points

    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homog = np.hstack([points.astype(np.float32), ones])
    mapped = (transform @ homog.T).T
    w = mapped[:, 2:3]
    w = np.where(np.abs(w) < 1e-8, 1.0, w)
    return mapped[:, :2] / w


def _metadata_transform(result: VisualizationResult, key: str) -> np.ndarray:
    """Read a 3x3 transform matrix from result metadata."""
    raw = result.metadata.get(key)
    if raw is None:
        return np.eye(3, dtype=np.float32)
    try:
        mat = np.asarray(raw, dtype=np.float32)
        if mat.shape == (3, 3):
            return mat
    except Exception:
        pass
    return np.eye(3, dtype=np.float32)


def _display_image_and_keypoints(
    result: VisualizationResult,
    image_data,
    keypoints: List[Keypoint],
    image_mode: str,
    transform_key: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare display image and keypoints in that image coordinate frame.

    Keypoints are stored in processed-image coordinates. For original mode, they
    are mapped back through inverse(original->processed) transform, then scaled
    to the display size.
    """
    if image_data is None:
        return np.zeros((64, 64, 3), dtype=np.uint8), np.empty((0, 2), dtype=np.float32)

    original = _to_bgr(image_data.original)
    processed = _to_bgr(image_data.processed)

    if processed is not None and processed.size > 0:
        target_h, target_w = processed.shape[:2]
    elif original is not None and original.size > 0:
        target_h, target_w = original.shape[:2]
    else:
        target_h, target_w = 64, 64

    base_pts = np.array([[float(kp.x), float(kp.y)] for kp in keypoints], dtype=np.float32) if keypoints else np.empty((0, 2), dtype=np.float32)

    mode = (image_mode or "p").lower()
    if mode == "o":
        src = original if original is not None and original.size > 0 else processed
        if src is None or src.size == 0:
            src = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        transform_orig_to_processed = _metadata_transform(result, transform_key)
        try:
            transform_processed_to_orig = np.linalg.inv(transform_orig_to_processed).astype(np.float32)
        except np.linalg.LinAlgError:
            transform_processed_to_orig = np.eye(3, dtype=np.float32)

        pts_in_orig = _apply_transform_to_points(base_pts, transform_processed_to_orig)

        src_h, src_w = src.shape[:2]
        sx = float(target_w) / float(max(1, src_w))
        sy = float(target_h) / float(max(1, src_h))
        pts_display = pts_in_orig.copy()
        if pts_display.size > 0:
            pts_display[:, 0] *= sx
            pts_display[:, 1] *= sy
    else:
        src = processed if processed is not None and processed.size > 0 else original
        if src is None or src.size == 0:
            src = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        src_h, src_w = src.shape[:2]
        sx = float(target_w) / float(max(1, src_w))
        sy = float(target_h) / float(max(1, src_h))
        pts_display = base_pts.copy()
        if pts_display.size > 0:
            pts_display[:, 0] *= sx
            pts_display[:, 1] *= sy

    if src.shape[0] != target_h or src.shape[1] != target_w:
        src = cv2.resize(src, (target_w, target_h), interpolation=cv2.INTER_AREA)

    return src, pts_display


def _draw_match_panel(result: VisualizationResult, viz_mode: str = "m", image_mode: str = "p") -> Tuple[np.ndarray, int]:
    """Draw image pair with match overlays.
    
    Args:
        result: VisualizationResult containing images, keypoints, matches
        viz_mode: Visualization mode: 'm' (matches), 'k' (keypoints), 'b' (both)
        image_mode: Image mode: 'o' (original), 'p' (processed)
    """
    img1, pts1 = _display_image_and_keypoints(
        result,
        result.image1,
        result.keypoints1,
        image_mode,
        "visualization_transform_img1_orig_to_processed",
    )
    img2, pts2 = _display_image_and_keypoints(
        result,
        result.image2,
        result.keypoints2,
        image_mode,
        "visualization_transform_img2_orig_to_processed",
    )

    if img1.shape[0] != img2.shape[0]:
        target_h = max(img1.shape[0], img2.shape[0])
        old_w1 = max(1, img1.shape[1])
        old_h1 = max(1, img1.shape[0])
        old_w2 = max(1, img2.shape[1])
        old_h2 = max(1, img2.shape[0])

        img1 = cv2.resize(img1, (int(img1.shape[1] * (target_h / img1.shape[0])), target_h), interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, (int(img2.shape[1] * (target_h / img2.shape[0])), target_h), interpolation=cv2.INTER_AREA)

        if pts1.size > 0:
            pts1[:, 0] *= float(img1.shape[1]) / float(old_w1)
            pts1[:, 1] *= float(img1.shape[0]) / float(old_h1)
        if pts2.size > 0:
            pts2[:, 0] *= float(img2.shape[1]) / float(old_w2)
            pts2[:, 1] *= float(img2.shape[0]) / float(old_h2)

    kpts1 = _cv2_keypoints([Keypoint(x=float(x), y=float(y)) for x, y in pts1])
    kpts2 = _cv2_keypoints([Keypoint(x=float(x), y=float(y)) for x, y in pts2])

    vis = cv2.hconcat([img1.copy(), img2.copy()])
    left_width = img1.shape[1]
    overlay = vis.copy()

    # Mode: 'k' - keypoints only
    if viz_mode == 'k':
        _draw_keypoints_overlay(
            overlay,
            kpts1,
            kpts2,
            left_width,
            list(range(len(kpts1))),
            list(range(len(kpts2))),
        )
        return _blend_annotations(vis, overlay), left_width
    
    selected_matches, _, _, _ = _selected_matches(result)

    has_inlier_info = False
    valid_matches: List[Match] = []

    for i, m in enumerate(selected_matches):
        if m.kp1_idx < 0 or m.kp2_idx < 0 or m.kp1_idx >= len(kpts1) or m.kp2_idx >= len(kpts2):
            continue
        valid_matches.append(m)
        if m.is_inlier is None:
            continue
        else:
            has_inlier_info = True

    # Draw match lines directly onto a shared base canvas to preserve image tones.
    for m in valid_matches:
        kp1 = kpts1[int(m.kp1_idx)]
        kp2 = kpts2[int(m.kp2_idx)]

        p1 = (int(kp1.pt[0]), int(kp1.pt[1]))
        p2 = (int(kp2.pt[0]) + left_width, int(kp2.pt[1]))

        if has_inlier_info:
            line_color = (0, 255, 0) if bool(m.is_inlier) else (0, 0, 255)
        else:
            line_color = (255, 255, 0)

        cv2.line(overlay, p1, p2, line_color, 1, cv2.LINE_AA)
        cv2.circle(overlay, p1, 2, line_color, -1)
        cv2.circle(overlay, p2, 2, line_color, -1)

    # Mode: 'b' - both matches and unmatched keypoints
    if viz_mode == 'b':
        unmatched_kp1, unmatched_kp2 = _get_unmatched_keypoints(result, selected_matches)

        _draw_keypoints_overlay(overlay, kpts1, kpts2, left_width, unmatched_kp1, unmatched_kp2)

    return _blend_annotations(vis, overlay), left_width


def render_match_visualization(result: VisualizationResult, viz_mode: str = "m", image_mode: str = "p") -> np.ndarray:
    """Render a single-image-pair matching summary image from VisualizationResult.
    
    Args:
        result: VisualizationResult containing images, keypoints, matches
        viz_mode: Visualization mode: 'm' (matches), 'k' (keypoints), 'b' (both)
        image_mode: Image mode: 'o' (original), 'p' (processed)
    """
    main_panel, left_img_w = _draw_match_panel(result, viz_mode=viz_mode, image_mode=image_mode)
    h, w = main_panel.shape[:2]

    canvas_w = max(w, MIN_CANVAS_WIDTH)
    if canvas_w > w:
        pad_left = (canvas_w - w) // 2
        pad_right = canvas_w - w - pad_left
        main_panel = cv2.copyMakeBorder(
            main_panel,
            0,
            0,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
    else:
        pad_left = 0

    split_x = pad_left + left_img_w
    right_img_w = max(1, w - left_img_w)

    header_h = 64
    file_strip_h = 34
    footer_h = 112

    header = np.full((header_h, canvas_w, 3), 235, dtype=np.uint8)
    file_strip = np.full((file_strip_h, canvas_w, 3), 218, dtype=np.uint8)
    footer = np.full((footer_h, canvas_w, 3), 28, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Header text
    file1 = result.image1.filename if result.image1 and result.image1.filename else "image1"
    file2 = result.image2.filename if result.image2 and result.image2.filename else "image2"

    # Map viz_mode to display text
    mode_display = {
        'm': 'Matches',
        'k': 'Keypoints',
        'b': 'Matches + Keypoints'
    }.get(viz_mode, viz_mode)

    image_mode_display = {
        'o': 'Original',
        'p': 'Processed',
    }.get((image_mode or 'p').lower(), image_mode)

    matcher_text = _truncate_to_pixel_width(
        f"Matcher: {result.method_name} | Viz: {mode_display} | Image: {image_mode_display}",
        int(canvas_w * 0.72),
        font,
        0.62,
        2,
    )
    modality_text = _truncate_to_pixel_width(
        f"Modality: {result.modality or 'N/A'}",
        int(canvas_w * 0.72),
        font,
        0.55,
        1,
    )
    cv2.putText(header, matcher_text, (10, 24), font, 0.62, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(header, modality_text, (10, 49), font, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # Draw filenames directly above corresponding image regions.
    max_left_name_w = max(40, left_img_w - 20)
    max_right_name_w = max(40, right_img_w - 20)
    file1_txt = _truncate_to_pixel_width(file1, max_left_name_w, font, 0.52, 1)
    file2_txt = _truncate_to_pixel_width(file2, max_right_name_w, font, 0.52, 1)

    file1_w = cv2.getTextSize(file1_txt, font, 0.52, 1)[0][0]
    file2_w = cv2.getTextSize(file2_txt, font, 0.52, 1)[0][0]

    left_center_x = pad_left + (left_img_w // 2)
    right_center_x = split_x + (right_img_w // 2)

    file1_x = max(6, left_center_x - (file1_w // 2))
    file2_x = min(canvas_w - file2_w - 6, max(split_x + 6, right_center_x - (file2_w // 2)))

    cv2.putText(file_strip, file1_txt, (file1_x, 24), font, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(file_strip, file2_txt, (file2_x, 24), font, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.line(file_strip, (split_x, 0), (split_x, file_strip_h - 1), (175, 175, 175), 1)

    # Footer text
    total = int(result.metadata.get("visualization_total_matches", len(result.matches)))
    drawn = int(result.metadata.get("visualization_drawn_matches", len(result.matches)))

    pred = result.is_same_person_pred
    gt = result.ground_truth
    is_correct = result.is_correct

    if is_correct is None:
        pred_color = (230, 230, 230)
    else:
        pred_color = (0, 220, 0) if is_correct else (0, 0, 255)

    line1 = (
        f"Kpts: img1={len(result.keypoints1)} img2={len(result.keypoints2)} | "
        f"Matches: total={total} drawn={drawn}"
    )
    line2 = (
        f"Inliers={result.num_inliers} | Inlier ratio={result.inlier_ratio:.3f} | "
        f"Reproj err={f'{float(result.reprojection_error):.4f} px' if result.reprojection_error is not None else 'N/A'}"
    )

    line1 = _truncate_to_pixel_width(line1, canvas_w - 20, font, 0.55, 1)
    line2 = _truncate_to_pixel_width(line2, canvas_w - 20, font, 0.55, 1)

    cv2.putText(footer, line1, (10, 28), font, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(footer, line2, (10, 56), font, 0.55, (235, 235, 235), 1, cv2.LINE_AA)

    seg_pred = f"Prediction={pred}"
    seg_conf = f"Confidence={result.verification_confidence:.3f}"
    seg_gt_correct = f"GT={gt} | Correct={is_correct}"
    _draw_segmented_text(
        footer,
        10,
        92,
        font,
        0.58,
        [
            (seg_pred, pred_color, 2),
            (" | ", (235, 235, 235), 1),
            (seg_conf, (235, 235, 235), 2),
            (" | ", (235, 235, 235), 1),
            (seg_gt_correct, (235, 235, 235), 1),
        ],
    )

    return cv2.vconcat([header, file_strip, main_panel, footer])


def save_visualization_image(vis_img: np.ndarray, output_path: str) -> str:
    """Save visualization image to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), vis_img)
    if not ok:
        raise RuntimeError(f"Failed to write visualization image: {path}")
    return str(path)


def show_visualization_image(vis_img: np.ndarray, title: str = "Match Visualization") -> None:
    """Display visualization image with matplotlib if available."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Warning: matplotlib is unavailable, cannot display visualization window.")
        return

    plt.figure(figsize=(16, 9))
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
