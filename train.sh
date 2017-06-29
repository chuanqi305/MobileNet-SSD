#!/bin/sh
../../build/tools/caffe train -solver="solver.prototxt" \
-weights="MobileNetSSD_train.caffemodel"
-gpu 0 
