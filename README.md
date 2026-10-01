# YOLOv5 VisDrone deployment on RDK X5

This repository records a YOLOv5s deployment workflow for the VisDrone2019-DET dataset and the Horizon Robotics RDK X5 (Bayes-e BPU).

## Scope and attribution

- The training framework and base model come from [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5), released under AGPL-3.0.
- The project-specific work is the VisDrone configuration, deployment-oriented Detect export, calibration-data preparation, RDK X5 mapper configuration, and board-side output contract.
- This is a deployment adaptation, not a from-scratch implementation of YOLOv5.

The repository name contains the historical typo `delect`; it means `detect`.

## Deployment flow

```text
VisDrone2019-DET
      ↓
YOLOv5s training / validation
      ↓
ONNX export with three raw NHWC detection-head outputs
      ↓
Representative RGB float32 calibration data (0..255)
      ↓
hb_mapper PTQ compile for Bayes-e
      ↓
RDK X5 NV12 input → BPU inference → sigmoid/decode/NMS on host
```

## Project-specific files

| File                 | Purpose                                                                                                                                     |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `data/VisDrone.yaml` | VisDrone2019-DET paths and 10 classes                                                                                                       |
| `models/yolo.py`     | Export branch: detection-head conv output is transposed from NCHW to NHWC; sigmoid/grid/anchor decode is left to board-side post-processing |
| `preprocess_640.py`  | Generates representative 640×640 RGB/NCHW float32 calibration tensors                                                                       |
| `rdk_x5_config.yaml` | Bayes-e, NV12 runtime input, PTQ calibration and latency-oriented compile settings                                                          |

## Prepare calibration data

Calibration files and compiled models are generated artifacts and are intentionally not committed.

```bash
python preprocess_640.py \
  --src /data/datasets/prepare_data \
  --dst /data/yolov5/calibration_data_rgb_f32 \
  --limit 200
```

The script writes RGB/NCHW `float32` values in the `0..255` range. `rdk_x5_config.yaml` applies `data_scale=1/255`; do not divide the calibration tensors by 255 a second time.

Use representative images covering target scale, density, lighting and viewpoints. Calibration data selects quantization ranges; a separate labeled validation split is still required to measure accuracy.

## Train and export

```bash
python train.py --data data/VisDrone.yaml --weights yolov5s.pt --img 640
python export.py --weights runs/train/exp/weights/best.pt --include onnx --imgsz 640 --opset 11
```

Choose the ONNX opset according to both the installed exporter and the RDK toolchain support matrix. A higher opset is not automatically better.

## Compile for RDK X5

Update the absolute paths in `rdk_x5_config.yaml`, then run the mapper in the official RDK X5 conversion environment:

```bash
hb_mapper makertbin --model-type onnx --config rdk_x5_config.yaml
```

The expected artifact name is `yolov5s_visdrone_640x640_nv12.bin`. Publish large model artifacts through a versioned release with a checksum instead of committing them to Git.

## Output contract

For detection export, each head returns a raw NHWC tensor:

```text
[batch, height, width, anchors × (classes + 5)]
```

For VisDrone (`classes=10`, three anchors per scale), the last dimension is `3 × 15 = 45`. Board-side post-processing must apply sigmoid, grid/anchor/stride decode, confidence filtering, coordinate restoration and class-aware NMS.

## Validation checklist

Use the same preprocessed input when comparing stages:

1. Check ONNX with `onnx.checker` and confirm all output names and shapes.
2. Compare PyTorch export-branch and ONNX tensors before quantization.
3. Compare ONNX and BPU tensor distributions or cosine similarity after accounting for quantization.
4. Run one shared decoder and compare boxes, scores and classes.
5. Report validation-set `mAP50`, `mAP50-95`, BPU P50/P95 latency, end-to-end FPS, memory and temperature with the test conditions.

No accuracy or latency number is claimed in this repository until a reproducible report is committed.

## Known limitations

- The repository currently contains a full upstream YOLOv5 snapshot, so project-specific changes should be reviewed through the file list above.
- Small-object accuracy is not proven by successful compilation. Evaluate by object-size slices before changing input resolution, adding a P2 head, tiled inference or quantization strategy.
- NV12 runtime correctness depends on color order, range, stride and resize behavior matching the mapper configuration.

## License

The upstream YOLOv5 code and this adaptation remain under the repository's AGPL-3.0 license. VisDrone is governed by its own dataset terms.
