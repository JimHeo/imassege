# imassege

**Image Semantic Segmentation Example**

## Environments

### Specification

- Ubuntu 18.04
- NVIDIA RTX 3090
- CUDA 11.1
- CuDNN 8.1.0

### Python Dependencies

- python 3.8.12
- pytorch 1.9.1
- opencv-python 4.5.4.58
- albumentations 1.1.0
- numpy 1.19.2
- tqdm 4.62.3
- matplotlib 3.2.2

## Directory Tree

```
imassege
├──dataset/
|    ├── your-own-dataset/
├──dcnn_module/
|    ├── neural_network/
|    |    ├── deeplab/
|    |    |    ├── backbone/
|    |    |    |    ├── __init__.py
|    |    |    |    ├── drn.py
|    |    |    |    ├── mobilenet.py
|    |    |    |    ├── resnet.py
|    |    |    |    ├── xception.py
|    |    |    ├── decoder/
|    |    |    |    ├── __init__.py
|    |    |    |    ├── deeplab_decoder.py
|    |    |    |    ├── unet_decoder.py
|    |    |    ├── sync_batchnorm/
|    |    |    |    ├── __init__.py
|    |    |    |    ├── batchnorm.py
|    |    |    |    ├── comm.py
|    |    |    |    ├── replicate.py
|    |    |    |    ├── unittest.py
|    |    |    ├── __init__.py
|    |    |    ├── aspp.py
|    |    |    ├── deeplab.py
|    |    ├── unet.py
|    |    ├── plainUnet.py
|    |    ├── densePlainUnet.py
|    |    ├── mini_unet.py
|    ├── config.py
|    ├── dataset.py
|    ├── utils/
|    |    ├── metrics_numpy.py
|    |    ├── metrics_pytorch.py
|    |    ├── preprocessing.py
├──output/
|    ├── predictions/
├──train.py
├──predict.py
├──README.md
```
