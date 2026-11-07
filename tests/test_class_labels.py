"""Визначення класу за назвами папок."""

from cxr_xai.data import class_labels


def test_infer_class_from_folders():
    from pathlib import Path
    assert class_labels.infer_class(Path("train/COVID/img1.png")) == "COVID-19"
    assert class_labels.infer_class(Path("Lung_Opacity/x.jpg")) == "Lung Opacity"
    assert class_labels.infer_class(Path("test/tuberculosis/sub/x.png")) == "Tuberculosis"
    assert class_labels.infer_class(Path("misc/x.png")) is None
    assert class_labels.infer_source_split(Path("Train/Normal/x.png")) == "train"
