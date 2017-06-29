import numpy as np  
import sys,os  
from scipy import misc
import cv2
caffe_root = '../../'
sys.path.insert(0, caffe_root + 'python')  
import caffe  

  
net_file= 'MobileNet_deploy.prototxt'  
caffe_model='MobileNet_deploy.caffemodel'  
net = caffe.Net(net_file,caffe_model,caffe.TEST)  

img = cv2.imread('images/000067.jpg')
img = cv2.resize(img, (224,224))

img = (img / 255. - 0.5) * 2.0

img = img.astype(np.float32)
img = img.transpose((2, 0, 1))


net.blobs['data'].data[...] = img

out = net.forward()  

output_prob = out['fc'].reshape((1001))
idx = np.argmax(output_prob)
print(idx)
print(output_prob[idx])
