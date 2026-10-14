"""Kiểm chứng thang tọa độ trên 9 box đã in ở cell YOLO → JSON.

Đây là kiểm tra hình học có kiểm soát, KHÔNG phải điểm model trên 311 ảnh test.
Ground truth tham chiếu được dựng từ các box JSON đã làm tròn trong output.
Chạy từ repo: python report/coordinate_sanity.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.score import evaluate  # noqa: E402

BOXES = [
    {"label": "crack", "bbox_2d": [430, 467, 583, 506]},
    {"label": "pothole", "bbox_2d": [361, 475, 414, 522]},
    {"label": "manhole", "bbox_2d": [317, 481, 616, 625]},
    {"label": "pothole", "bbox_2d": [583, 500, 627, 614]},
    {"label": "crack", "bbox_2d": [580, 508, 655, 614]},
    {"label": "pothole", "bbox_2d": [259, 517, 311, 572]},
    {"label": "crack", "bbox_2d": [397, 628, 447, 717]},
    {"label": "crack", "bbox_2d": [339, 683, 398, 808]},
    {"label": "crack", "bbox_2d": [158, 789, 361, 994]},
]


def main():
    width, height = 640, 360
    objects = []
    pixels = []
    for box in BOXES:
        x1, y1, x2, y2 = box["bbox_2d"]
        objects.append({"label": box["label"], "yolo": [
            (x1 + x2) / 2000, (y1 + y2) / 2000,
            (x2 - x1) / 1000, (y2 - y1) / 1000]})
        pixels.append({"label": box["label"], "bbox_2d": [
            x1 * width / 1000, y1 * height / 1000,
            x2 * width / 1000, y2 * height / 1000]})
    entry = {"image_id": "reference_train_sample_0", "width": width,
             "height": height, "objects": objects}
    cases = {
        "norm1000_scored_as_norm1000": (BOXES, "norm1000"),
        "norm1000_scored_as_pixel": (BOXES, "pixel"),
        "pixel_scored_as_pixel": (pixels, "pixel"),
        "pixel_scored_as_norm1000": (pixels, "norm1000"),
    }
    results = {}
    for name, (boxes, coord) in cases.items():
        metrics = evaluate([entry], {entry["image_id"]: json.dumps(boxes)},
                           coord=coord, n_boot=40, seed=0)
        results[name] = {"micro": metrics["micro"],
                         "parse_valid_rate": metrics["parse_valid_rate"]}
        print(name, "F1=", metrics["micro"]["f1"],
              "TP/FP/FN=", tuple(metrics["micro"][k] for k in ("tp", "fp", "fn")))
    result = {"method": "controlled_geometry_check_from_rounded_training_answer",
              "source": "Colab output of YOLO-to-JSON preview cell, first train sample",
              "is_full_test_model_evaluation": False, "n_images": 1,
              "n_reference_boxes": len(BOXES), "width": width, "height": height,
              "reference_boxes": BOXES, "results": results}
    Path(__file__).with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
