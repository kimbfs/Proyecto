# TF Keras YOLOv4/v3/v2 Modelset

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)



Download or clone the official repository (tested on d38c3d8 commit).
Use the following commands to get original model (named yolov3 in repository) and convert it to Keras* format (see details in the README.md file in the official repository):
Download YOLO v3 weights: ``` wget -O weights/yolov3.weights https://pjreddie.com/media/files/yolov3.weights ```
Convert model weights to Keras*: ``` python tools/model_converter/convert.py cfg/yolov3.cfg weights/yolov3.weights weights/yolov3.h5 ```
Convert model to protobuf: ``` python tools/model_converter/keras_to_tensorflow.py –input_model weights/yolov3.h5 –output_model=weights/yolo-v3.pb ```
## Quick Start

1. Install requirements on Ubuntu 16.04/18.04:

```
# apt install python3-opencv
# pip install Cython
# pip install -r requirements.txt
```

### Train
1. Generate train/val/test annotation file and class names file.

    Data annotation file format:
    * One row for one image in annotation file;
    * Row format: `image_file_path box1 box2 ... boxN`;
    * Box format: `x_min,y_min,x_max,y_max,class_id` (no space).
    * Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```
    1. For VOC style dataset, you can use [voc_annotation.py](https://github.com/david8862/keras-YOLOv3-model-set/blob/master/tools/dataset_converter/voc_annotation.py) to convert original dataset to our annotation file:
       ```
       # cd tools/dataset_converter/ && python voc_annotation.py -h
       usage: voc_annotation.py [-h] [--dataset_path DATASET_PATH] [--year YEAR]
                                [--set SET] [--output_path OUTPUT_PATH]
                                [--classes_path CLASSES_PATH] [--include_difficult]
                                [--include_no_obj]

       convert PascalVOC dataset annotation to txt annotation file

       optional arguments:
         -h, --help            show this help message and exit
         --dataset_path DATASET_PATH
                               path to PascalVOC dataset, default is ../../VOCdevkit
         --year YEAR           subset path of year (2007/2012), default will cover
                               both
         --set SET             convert data set, default will cover train, val and
                               test
         --output_path OUTPUT_PATH
                               output path for generated annotation txt files,
                               default is ./
         --classes_path CLASSES_PATH
                               path to class definitions
         --include_difficult   to include difficult object
         --include_no_obj      to include no object image
       ```
       By default, the VOC convert script will try to go through both VOC2007/VOC2012 dataset dir under the dataset_path and generate train/val/test annotation file separately, like:
       ```
       2007_test.txt  2007_train.txt  2007_val.txt  2012_train.txt  2012_val.txt
       ```
       You can merge these train & val annotation file as your need. For example, following cmd will creat 07/12 combined trainval dataset:
       ```
       # cp 2007_train.txt trainval.txt
       # cat 2007_val.txt >> trainval.txt
       # cat 2012_train.txt >> trainval.txt
       # cat 2012_val.txt >> trainval.txt
       ```
       P.S. You can use [LabelImg](https://github.com/tzutalin/labelImg) to annotate your object detection dataset with Pascal VOC XML format

    2. For COCO style dataset, you can use [coco_annotation.py](https://github.com/david8862/keras-YOLOv3-model-set/blob/master/tools/dataset_converter/coco_annotation.py) to convert original dataset to our annotation file:
       ```
       # cd tools/dataset_converter/ && python coco_annotation.py -h
       usage: coco_annotation.py [-h] [--dataset_path DATASET_PATH]
                                 [--output_path OUTPUT_PATH]
                                 [--classes_path CLASSES_PATH] [--include_no_obj]
                                 [--customize_coco]

       convert COCO dataset annotation to txt annotation file

       optional arguments:
         -h, --help            show this help message and exit
         --dataset_path DATASET_PATH
                               path to MSCOCO dataset, default is ../../mscoco2017
         --output_path OUTPUT_PATH
                               output path for generated annotation txt files,
                               default is ./
         --classes_path CLASSES_PATH
                               path to class definitions, default is
                               ../configs/coco_classes.txt
         --include_no_obj      to include no object image
         --customize_coco      It is a user customize coco dataset. Will not follow
                               standard coco class label
       ```
       This script will try to convert COCO instances_train2017 and instances_val2017 under dataset_path. You can change the code for your dataset

2. [train.py](https://github.com/david8862/keras-YOLOv3-model-set/blob/master/train.py)
```
# python train.py -h
usage: train.py [-h] [--model_type MODEL_TYPE] [--anchors_path ANCHORS_PATH]
                [--model_image_size MODEL_IMAGE_SIZE]
                [--weights_path WEIGHTS_PATH]
                [--annotation_file ANNOTATION_FILE]
                [--val_annotation_file VAL_ANNOTATION_FILE]
                [--val_split VAL_SPLIT] [--classes_path CLASSES_PATH]
                [--batch_size BATCH_SIZE] [--optimizer {adam,rmsprop,sgd}]
                [--learning_rate LEARNING_RATE]
                [--decay_type {None,cosine,exponential,polynomial,piecewise_constant}]
                [--transfer_epoch TRANSFER_EPOCH]
                [--freeze_level {None,0,1,2}] [--init_epoch INIT_EPOCH]
                [--total_epoch TOTAL_EPOCH] [--multiscale]
                [--rescale_interval RESCALE_INTERVAL]
                [--enhance_augment {None,mosaic}]
                [--label_smoothing LABEL_SMOOTHING] [--multi_anchor_assign]
                [--elim_grid_sense] [--data_shuffle] [--gpu_num GPU_NUM]
                [--model_pruning] [--eval_online]
                [--eval_epoch_interval EVAL_EPOCH_INTERVAL]
                [--save_eval_checkpoint]

optional arguments:
  -h, --help            show this help message and exit
  --model_type MODEL_TYPE
                        YOLO model type: yolo3_mobilenet_lite/tiny_yolo3_mobil
                        enet/yolo3_darknet/..., default=yolo3_mobilenet_lite
  --anchors_path ANCHORS_PATH
                        path to anchor definitions,
                        default=configs/yolo3_anchors.txt
  --model_image_size MODEL_IMAGE_SIZE
                        Initial model image input size as <height>x<width>,
                        default=416x416
  --weights_path WEIGHTS_PATH
                        Pretrained model/weights file for fine tune
  --annotation_file ANNOTATION_FILE
                        train annotation txt file, default=trainval.txt
  --val_annotation_file VAL_ANNOTATION_FILE
                        val annotation txt file, default=None
  --val_split VAL_SPLIT
                        validation data persentage in dataset if no val
                        dataset provide, default=0.1
  --classes_path CLASSES_PATH
                        path to class definitions,
                        default=configs/voc_classes.txt
  --batch_size BATCH_SIZE
                        Batch size for train, default=16
  --optimizer {adam,rmsprop,sgd}
                        optimizer for training (adam/rmsprop/sgd),
                        default=adam
  --learning_rate LEARNING_RATE
                        Initial learning rate, default=0.001
  --decay_type {None,cosine,exponential,polynomial,piecewise_constant}
                        Learning rate decay type, default=None
  --transfer_epoch TRANSFER_EPOCH
                        Transfer training (from Imagenet) stage epochs,
                        default=20
  --freeze_level {None,0,1,2}
                        Freeze level of the model in transfer training stage.
                        0:NA/1:backbone/2:only open prediction layer
  --init_epoch INIT_EPOCH
                        Initial training epochs for fine tune training,
                        default=0
  --total_epoch TOTAL_EPOCH
                        Total training epochs, default=250
  --multiscale          Whether to use multiscale training
  --rescale_interval RESCALE_INTERVAL
                        Number of iteration(batches) interval to rescale input
                        size, default=10
  --enhance_augment {None,mosaic}
                        enhance data augmentation type (None/mosaic),
                        default=None
  --label_smoothing LABEL_SMOOTHING
                        Label smoothing factor (between 0 and 1) for
                        classification loss, default=0
  --multi_anchor_assign
                        Assign multiple anchors to single ground truth
  --elim_grid_sense     Eliminate grid sensitivity
  --data_shuffle        Whether to shuffle train/val data for cross-validation
  --gpu_num GPU_NUM     Number of GPU to use, default=1
  --model_pruning       Use model pruning for optimization, only for TF 1.x
  --eval_online         Whether to do evaluation on validation dataset during
                        training
  --eval_epoch_interval EVAL_EPOCH_INTERVAL
                        Number of iteration(epochs) interval to do evaluation,
                        default=10
  --save_eval_checkpoint
                        Whether to save checkpoint with best evaluation result
```

**NOTE**: if enable "--elim_grid_sense" feature during training, recommended to also use it in following demo/inference step.

Following is a reference training config cmd:
```
# python train.py --model_type=yolo3_mobilenet_lite --anchors_path=configs/yolo3_anchors.txt --annotation_file=trainval.txt --classes_path=configs/voc_classes.txt --eval_online --save_eval_checkpoint
```

# Citation
Please cite keras-YOLOv3-model-set in your publications if it helps your research:
```
@article{MobileNet-Yolov3,
     Author = {Adam Yang},
     Year = {2018}
}
@article{keras-yolo3,
     Author = {qqwweee},
     Year = {2018}
}
@article{YAD2K,
     title={YAD2K: Yet Another Darknet 2 Keras},
     Author = {allanzelener},
     Year = {2017}
}
@article{yolov4,
     title={YOLOv4: Optimal Speed and Accuracy of Object Detection},
     author={Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao},
     journal = {arXiv},
     year={2020}
}
@article{yolov3,
     title={YOLOv3: An Incremental Improvement},
     author={Redmon, Joseph and Farhadi, Ali},
     journal = {arXiv},
     year={2018}
}
@article{redmon2016yolo9000,
  title={YOLO9000: Better, Faster, Stronger},
  author={Redmon, Joseph and Farhadi, Ali},
  journal={arXiv preprint arXiv:1612.08242},
  year={2016}
}
@article{Focal Loss,
     title={Focal Loss for Dense Object Detection},
     author={Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár},
     journal = {arXiv},
     year={2017}
}
@article{GIoU,
     title={Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression},
     author={Hamid Rezatofighi, Nathan Tsoi1, JunYoung Gwak1, Amir Sadeghian, Ian Reid, Silvio Savarese},
     journal = {arXiv},
     year={2019}
}
@article{DIoU Loss,
     title={Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression},
     author={Zhaohui Zheng, Ping Wang, Wei Liu, Jinze Li, Rongguang Ye, Dongwei Ren},
     journal = {arXiv},
     year={2020}
}
@inproceedings{tide-eccv2020,
  author    = {Daniel Bolya and Sean Foley and James Hays and Judy Hoffman},
  title     = {TIDE: A General Toolbox for Identifying Object Detection Errors},
  booktitle = {ECCV},
  year      = {2020},
}
```
