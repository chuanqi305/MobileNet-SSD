# MobileNet-SSD
A caffe implementation of Google MobileNet SSD detection network, with pretrained weights on VOC0712.

### Run
1. Download [SSD](https://github.com/weiliu89/caffe/tree/ssd) source code and compile (follow the SSD README).
2. Put all the files in SSD_HOME/examples/
3. Run merge_bn.py to generate deploy caffemodel.
4. Run demo.py to test the detection result.


### Train your own dataset
1. Convert your own dataset to lmdb database (follow the SSD README).
2. Modify the MobileNetSSD_train.prototxt like this (or use gen.py):
  * Change the lmdb database and labelmap file path.
```
  data_param {
    source: "/home/yaochuanqi/data/VOCdevkit/VOC0712/lmdb/VOC0712_trainval_lmdb" # change to your lmdb path
    batch_size: 32
    backend: LMDB
  }

   ...

  label_map_file: "../../data/VOC0712/labelmap_voc.prototxt" # change to your labelmap file
```
  * Change the mbox_conf layer output num for all 5 mbox_conv layers

```
  convolution_param {
    num_output: 84 # 84 = 21 * 4, set to (classnum + 1) * 4 , "+1" is for background
    bias_term: false

  ...

  convolution_param {
    num_output: 126 # 126 = 21 * 6, set to (classnum + 1) * 6 , "+1" is for background
    bias_term: false
```
3. Run train.sh After about 30000 iteration, the loss should be 2.0 - 3.0.
4. Run merge_bn.py to generate your own deploy caffemodel.
     
### About some details
There are 3 differences between my model and [MobileNet-SSD on tensorflow](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md):
1. I replaced the tensorflow's ReLU6 layer with ReLU.
2. My batch normal eps=0.00001 vs tensorflow's eps=0.001.
3. For the conv11 anchors, I use [(0.2, 1.0), (0.2, 2.0), (0.2, 0.5)] vs tensorflow's [(0.1, 1.0), (0.2, 2.0), (0.2, 0.5)].

