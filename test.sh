#!/bin/sh
../../build/tools/caffe train -solver="solver_test.prototxt" \
--weights=snapshot/mobilenet_iter_73000.caffemodel \
-gpu 0
