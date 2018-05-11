# -*- coding: utf-8 -*
import numpy as np  
import sys,os  
import cv2

import caffe  
import time


net_file= 'faceboxes_deploy_4_2.prototxt'
caffe_model='faceboxes_iter_120000.caffemodel'
# image_dir = "fddb"
# image_list = "FDDB-list.txt"
# image_dir_out = "fddb_out"
image_dir = "xiaomi_faces_val_640x360"
image_list = "face_list.txt"
image_dir_out = "xiaomi_out"

# d_ret_file = "fddb_result.txt"
d_ret_file = "xiaomi_result.txt"

if not os.path.exists(image_dir):
    print("{} does not exist".format(image_dir))
    exit()

if not os.path.exists(image_dir_out):
    print("we create {} dir".format(image_dir_out))
    os.mkdir(image_dir_out)

if not os.path.exists(caffe_model):
    print("FaceBoxes_deploy.caffemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()

caffe.set_mode_gpu()
net = caffe.Net(net_file,caffe_model,caffe.TEST)  

CLASSES = ('background','face')

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel


def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect(img_dir, imgfile):

    full_path = os.path.join(img_dir, imgfile + ".jpg")
    frame = cv2.imread(full_path)
    transformed_image = transformer.preprocess('data', frame)
    # print img
    # print(transformed_image)
    net.blobs['data'].data[...] = transformed_image


    time_start=time.time()
    out = net.forward()
    time_end=time.time()
    print (time_end-time_start)
    #print(out['detection_out'])
    box, conf, cls = postprocess(frame, out)

    count = 0
    _str = ""
    str_name = imgfile + "\n"
    str_box = ""

    _str += str_name
    for i in range(len(box)):
        p1 = (box[i][0], box[i][1])
        p2 = (box[i][2], box[i][3])
        if conf[i] > 0.5 :
            cv2.rectangle(frame, p1, p2, (0,255,0))
            p3 = (max(p1[0], 15), max(p1[1], 15))
            title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
            cv2.putText(frame, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)

        str_box += str(box[i][0]) + " " \
             + str(box[i][1]) + " " \
             + str(box[i][2] - box[i][0]) + " " \
             + str(box[i][3] - box[i][1]) + " " \
             + str(conf[i]) + "\n"
        count += 1
    _str += str(count) + "\n"
    _str += str_box
    print(_str)
    return _str, frame

# for f in os.listdir(test_dir):
#     if detect(test_dir + "/" + f) == False:
#        break


imgs_path_fd = open(image_list, "r")
imgs_path = imgs_path_fd.readlines()
imgs_path_fd.close()

print(imgs_path)

str_ret =""
for img_path in imgs_path:
    _str, frame = detect(image_dir, img_path.strip("\n"))
    str_ret += _str
    cv2.imwrite(os.path.join(image_dir_out, img_path.replace("/","_").strip("\n") + ".jpg"), frame)

d_ret_fd = open(d_ret_file, "w")
d_ret_fd.writelines(str_ret)
d_ret_fd.close()
