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


def _select_image_source(image_data, image_mode: str) -> np.ndarray:
    """Select visualization image by mode: original ('o') or processed ('p')."""
    if image_data is None:
        return None

    mode = (image_mode or "p").lower()
    if mode == "o":
        if image_data.original is not None:
            return image_data.original
        return image_data.processed

    if image_data.processed is not None:
        return image_data.processed
    return image_data.original


def _draw_match_panel(result: VisualizationResult, viz_mode: str = "m", image_mode: str = "p") -> Tuple[np.ndarray, int]:
    """Draw image pair with match overlays.
    
    Args:
        result: VisualizationResult containing images, keypoints, matches
        viz_mode: Visualization mode: 'm' (matches), 'k' (keypoints), 'b' (both)
        image_mode: Image mode: 'o' (original), 'p' (processed)
    """
    img1_src = _select_image_source(result.image1, image_mode)
    img2_src = _select_image_source(result.image2, image_mode)

    img1 = _to_bgr(img1_src)
    img2 = _to_bgr(img2_src)
    if img1.shape[0] != img2.shape[0]:
        target_h = max(img1.shape[0], img2.shape[0])
        img1 = cv2.resize(img1, (int(img1.shape[1] * (target_h / img1.shape[0])), target_h), interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, (int(img2.shape[1] * (target_h / img2.shape[0])), target_h), interpolation=cv2.INTER_AREA)

    kpts1 = _cv2_keypoints(result.keypoints1)
    kpts2 = _cv2_keypoints(result.keypoints2)

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
