#!/bin/sh
../../build/tools/caffe train -solver="solver.prototxt" \
-weights="MobileNet_nofc.caffemodel" \
-gpu 0 
