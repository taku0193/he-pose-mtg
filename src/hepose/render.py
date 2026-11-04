"""Render named joints in original-image coordinates, without changing source frames."""

from hepose.schema import FrameResult
from hepose.video import VideoError, backend

EDGES = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("head_top", "head_bottom"),
    ("head_bottom", "left_shoulder"),
    ("head_bottom", "right_shoulder"),
)


def draw_poses(image, result: FrameResult, confidence_threshold: float = 0.0):
    if image.shape != (result.height, result.width, 3):
        raise VideoError("描画対象と推定結果の画像サイズが一致しません。")
    cv2 = backend()
    output = image.copy()
    for person in result.people:

        def clip(x, y):
            return (int(max(0, min(result.width - 1, x))), int(max(0, min(result.height - 1, y))))

        if person.bbox is not None:
            x1, y1, x2, y2 = person.bbox
            cv2.rectangle(output, clip(x1, y1), clip(x2, y2), (200, 120, 20), 1)
        points = {}
        for point in person.native_keypoints or person.keypoints:
            if point.x is None or not (
                0 <= point.x < result.width and 0 <= point.y < result.height
            ):
                continue
            if point.confidence is not None and point.confidence < confidence_threshold:
                continue
            points[point.name] = (int(round(point.x)), int(round(point.y)))
        for a, b in EDGES:
            if a in points and b in points:
                cv2.line(output, points[a], points[b], (0, 220, 80), 2)
        for position in points.values():
            cv2.circle(output, position, 2, (0, 180, 255), -1)
    label = {"success": "OK", "failed": "FAILED", "not_estimated": "NOT ESTIMATED"}[result.status]
    color = (0, 220, 80) if result.status == "success" else (0, 140, 255)
    cv2.putText(
        output,
        f"{result.frame_index}: {label}",
        (4, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        color,
        1,
        cv2.LINE_AA,
    )
    return output
