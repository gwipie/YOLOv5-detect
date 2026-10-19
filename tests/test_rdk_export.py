# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from models.yolo import Detect


def _detect_head():
    model = Detect(
        nc=1,
        anchors=((10, 13, 16, 30, 33, 23),) * 3,
        ch=(4, 4, 4),
    )
    model.stride = torch.tensor([8.0, 16.0, 32.0])
    model.eval()
    return model


def _features():
    return [torch.zeros(1, 4, size, size) for size in (8, 4, 2)]


def test_standard_export_keeps_decoded_detection_output():
    model = _detect_head()
    model.export = True

    output = model(_features())

    assert len(output) == 1
    assert output[0].shape == (1, 252, 6)


def test_rdk_x5_export_returns_raw_nhwc_detection_maps():
    model = _detect_head()
    model.export = True
    model.rdk_x5_export = True

    output = model(_features())

    assert len(output) == 3
    assert [tuple(item.shape) for item in output] == [
        (1, 8, 8, 18),
        (1, 4, 4, 18),
        (1, 2, 2, 18),
    ]
