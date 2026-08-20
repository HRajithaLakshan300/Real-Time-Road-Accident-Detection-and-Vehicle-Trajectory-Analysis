from src.geometry import bottom_center, iou


def test_bottom_center() -> None:
    assert bottom_center((10.0, 20.0, 30.0, 60.0)) == (20, 60)


def test_iou_identical_boxes() -> None:
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_disjoint_boxes() -> None:
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
