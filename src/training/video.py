#!/usr/bin/env python
# coding: utf-8

# In[1]:


from matplotlib import pyplot as plt
#get_ipython().magic(u'matplotlib notebook')

from keras.models import load_model
from model import get_personlab
from scipy.ndimage.filters import gaussian_filter
import cv2
import numpy as np
from time import time
from config import config
import random
from post_proc import *


# In[2]:


tic = time()
#model = get_personlab(train=False, with_preprocess_lambda=True,
#                      intermediate_supervision=True,
#                      intermediate_layer='res4b12_relu',
#                      build_base_func=get_resnet101_base,
#                      output_stride=16)
model = get_personlab(train=False, with_preprocess_lambda=True,
                      output_stride=8)
print 'Loading time: {}'.format(time()-tic)


# In[3]:


#model.load_weights('models/personlab_res101_400_r32_0510.h5')
model.load_weights('personlab_model_101_best.h5')


# In[4]:


# Pad image appropriately (to match relationship to output_stride as in training)
def pad_img(img, mult=16):
    h, w, _ = img.shape
    
    h_pad = 0
    w_pad = 0
    if (h-1)%mult > 0:
        h_pad = mult-((h-1)%mult)
    if (w-1)%mult > 0:
        w_pad = mult-((w-1)%mult)
    return np.pad(img, ((0,h_pad), (0,w_pad), (0,0)), 'constant')

#img = cv2.imread('testim.jpg')
#img = cv2.resize(img, (0,0), fx=.9, fy=.9)
#img = cv2.resize(img, (388,388))
#img = pad_img(img)
#print 'Image shape: {}'.format(img.shape)


# In[6]:


cap = cv2.VideoCapture('vid.mov')

while(cap.isOpened()):
    ret, frame = cap.read()

    img = cv2.resize(frame, (0,0), fx=.9, fy=.9)
    img = pad_img(img)
    
    outputs = model.predict(img[np.newaxis,...])
    outputs = [o[0] for o in outputs]

    H = compute_heatmaps(kp_maps=outputs[0], short_offsets=outputs[1])
    # Gaussian filtering helps when there are multiple local maxima for the same keypoint.
    for i in range(17):
        H[:,:,i] = gaussian_filter(H[:,:,i], sigma=2)
        
    pred_kp = get_keypoints(H)
    print(len(pred_kp))
    
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




