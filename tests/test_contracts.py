import json
import math
import unittest

from hepose.schema import (
    COCO17,
    FrameResult,
    Keypoint,
    Person,
    TemporalContext,
    coco17,
    restore_crop,
)
from hepose.timing import summarize, throughput


class ContractTests(unittest.TestCase):
    def test_unmapped_head_points_do_not_become_eyes(self):
        original = (Keypoint("nose", 10, 20), Keypoint("head_top", 8, 5))
        normalized = coco17(original)
        self.assertEqual(tuple(point.name for point in normalized), COCO17)
        self.assertEqual(normalized[0].x, 10)
        self.assertIsNone(normalized[1].x)
        self.assertIsNone(normalized[2].x)
        self.assertIsNone(normalized[1].confidence)
        self.assertEqual(original[1].name, "head_top")

    def test_crop_coordinates_restore_to_original_image(self):
        points = (Keypoint("left_wrist", 50, 25, 0.7), Keypoint("right_wrist"))
        restored = restore_crop(points, (100, 200, 300, 300), (100, 50))
        self.assertEqual((restored[0].x, restored[0].y), (200, 250))
        self.assertEqual(restored[0].confidence, 0.7)
        self.assertIsNone(restored[0].visibility)
        self.assertIsNone(restored[1].x)

    def test_zero_people_and_temporal_boundary_have_distinct_states(self):
        empty = FrameResult("run-1", "yolo26", 0, 0, 640, 480)
        boundary = FrameResult(
            "run-1",
            "tar-vitpose",
            0,
            0,
            640,
            480,
            status="not_estimated",
            reason="temporal_boundary",
        )
        self.assertEqual(json.loads(empty.to_json())["status"], "success")
        self.assertEqual(json.loads(boundary.to_json())["status"], "not_estimated")
        self.assertEqual(empty.people, ())

    def test_native_keypoints_and_null_tracking_id_survive_json(self):
        native = (Keypoint("head_bottom", 25, 40),)
        person = Person((0, 0, 100, 200), coco17(native), native_keypoints=native)
        result = FrameResult("run-1", "tar-vitpose", 2, 80, 640, 480, people=(person,))
        data = json.loads(result.to_json())
        self.assertIsNone(data["people"][0]["track_id"])
        self.assertEqual(data["people"][0]["native_keypoints"][0]["name"], "head_bottom")
        self.assertIsNone(data["people"][0]["keypoints"][1]["x"])

    def test_temporal_context_keeps_original_frame_numbers(self):
        context = TemporalContext(5, 1, 2)
        self.assertIsNone(context.references(0, 8))
        self.assertEqual(context.references(2, 8), (0, 1, 2, 3, 4))
        self.assertEqual(context.references(5, 8), (3, 4, 5, 6, 7))
        self.assertIsNone(context.references(6, 8))
        self.assertIsNone(context.references(1, 3))

    def test_invalid_coordinates_and_nan_are_rejected(self):
        for invalid in (math.nan, math.inf, -math.inf):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                Keypoint("nose", invalid, 1)
        with self.assertRaises(ValueError):
            Keypoint("nose", 1, None)
        with self.assertRaises(ValueError):
            Person((5, 0, 1, 10))
        with self.assertRaises(ValueError):
            coco17((Keypoint("nose"), Keypoint("nose")))

    def test_errors_require_a_reason_and_cannot_contain_people(self):
        with self.assertRaises(ValueError):
            FrameResult("r", "yolo26", 0, 0, 10, 10, status="failed")
        with self.assertRaises(ValueError):
            FrameResult(
                "r",
                "yolo26",
                0,
                0,
                10,
                10,
                status="failed",
                reason="inference",
                people=(Person((0, 0, 5, 5)),),
            )

    def test_statistics_preserve_missing_measurements(self):
        values = summarize([10, None, 20, 30])
        self.assertEqual(values["count"], 3)
        self.assertEqual(values["mean_ms"], 20)
        self.assertEqual(values["p50_ms"], 20)
        self.assertEqual(values["p95_ms"], 29)
        self.assertIsNone(summarize([None])["mean_ms"])
        self.assertEqual(throughput(10, 2000), 5)
        self.assertIsNone(throughput(10, 0))
        with self.assertRaises(ValueError):
            summarize([math.nan])


if __name__ == "__main__":
    unittest.main()
