import cv2
import numpy as np

from object_counter.counter import Detection


def detect_dark_objects(
    frame_bgr: np.ndarray,
    min_area: int = 800,
    label: str = "objeto escuro",
) -> list[Detection]:
    return detect_blobs_by_threshold(
        frame_bgr,
        threshold_mode=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        min_area=min_area,
        label=label,
    )


def detect_light_objects(
    frame_bgr: np.ndarray,
    min_area: int = 80,
    label: str = "objeto claro",
) -> list[Detection]:
    return detect_blobs_by_threshold(
        frame_bgr,
        threshold_mode=cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        min_area=min_area,
        label=label,
    )


def detect_contrast_objects(
    frame_bgr: np.ndarray,
    min_area: int = 120,
    label: str = "objeto",
) -> list[Detection]:
    dark = detect_dark_objects(frame_bgr, min_area=min_area, label=label)
    light = detect_light_objects(frame_bgr, min_area=min_area, label=label)
    return merge_overlapping_detections(dark + light)


def detect_blobs_by_threshold(
    frame_bgr: np.ndarray,
    threshold_mode: int,
    min_area: int,
    label: str,
) -> list[Detection]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    _, threshold = cv2.threshold(
        blurred,
        0,
        255,
        threshold_mode,
    )

    kernel = np.ones((5, 5), np.uint8)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _hierarchy = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    detections: list[Detection] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        detections.append(
            Detection(
                label=label,
                confidence=1.0,
                box=(x, y, width, height),
            )
        )

    return sorted(detections, key=lambda detection: (detection.box[1], detection.box[0]))


def merge_overlapping_detections(detections: list[Detection]) -> list[Detection]:
    kept: list[Detection] = []

    for detection in sorted(detections, key=lambda item: box_area(item.box), reverse=True):
        if any(intersection_over_union(detection.box, other.box) > 0.35 for other in kept):
            continue
        kept.append(detection)

    return sorted(kept, key=lambda item: (item.box[1], item.box[0]))


def box_area(box: tuple[int, int, int, int]) -> int:
    return box[2] * box[3]


def intersection_over_union(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second

    x_left = max(first_x, second_x)
    y_top = max(first_y, second_y)
    x_right = min(first_x + first_width, second_x + second_width)
    y_bottom = min(first_y + first_height, second_y + second_height)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    union = box_area(first) + box_area(second) - intersection
    return intersection / union
