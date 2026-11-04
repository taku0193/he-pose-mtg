"""Official TAR-ViTPose integration, with explicit native joints and original-frame windows."""

from hepose.schema import Keypoint, Person, coco17

NATIVE_JOINTS = (
    "nose",
    "head_bottom",
    "head_top",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
USED_JOINTS = (0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)


def decode_person(heatmaps, crop_xyxy, detector_bbox) -> Person:
    import numpy as np

    if (
        len(heatmaps.shape) != 4
        or heatmaps.shape[:2] != (1, 17)
        or not heatmaps.shape[2]
        or not heatmaps.shape[3]
    ):
        raise ValueError("TARのヒートマップは1×17×H×Wです。")
    if not np.isfinite(heatmaps).all():
        raise ValueError("TARのヒートマップに非有限値があります。")
    x1, y1, x2, y2 = crop_xyxy
    if not np.isfinite(crop_xyxy).all() or x2 <= x1 or y2 <= y1:
        raise ValueError("TARのクロップ領域が不正です。")
    height, width = heatmaps.shape[2:]
    native = []
    for index, name in enumerate(NATIVE_JOINTS):
        if index not in USED_JOINTS:
            native.append(Keypoint(name))
            continue
        position = int(heatmaps[0, index].argmax())
        x = x1 + (position % width) * (x2 - x1) / width
        y = y1 + (position // width) * (y2 - y1) / height
        native.append(Keypoint(name, float(x), float(y)))
    points = tuple(native)
    return Person(
        tuple(detector_bbox),
        coco17(points),
        native_keypoints=points,
        native_joint_definition="posetrack17_17slots",
        crop_xyxy=tuple(crop_xyxy),
    )


def validate_window(frames, target_frame_index):
    expected = list(range(target_frame_index - 2, target_frame_index + 3))
    if len(frames) != 5 or [frame.frame_index for frame in frames] != expected:
        raise ValueError("TARには中心の前後2枚を含む連続5フレームが必要です。")
    center = frames[2]
    if any(
        frame.width != center.width
        or frame.height != center.height
        or frame.image.shape != (center.height, center.width, 3)
        for frame in frames
    ):
        raise ValueError("TARの参照画像サイズが一致しません。")
    return center
