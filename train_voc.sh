#!/bin/sh
mkdir -p snapshot
../../build/tools/caffe train -solver="voc/solver.prototxt" \
-weights="mobilenet_iter_73000.caffemodel" \
-gpu 0 
