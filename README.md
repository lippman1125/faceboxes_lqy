Faceboxes Reproduce
===

1. CAFFE安装：<br>
---
Makefile.config已经修改了好了，使用GPU的方式 <br>
所以直接使用下面的命令编译： <br>
```C
    make -j8            
    # Make sure to include $CAFFE_ROOT/python to your PYTHONPATH. 
    make pycaffe        
    make test -j8       
    # (Optional)        
    make runtest -j8    
```
2. 数据处理：<br>
--- 
 >>(1) 利用脚本wider_face_2_voc.py脚本把wider_face数据转换成VOC格式。并遮盖掉小于20x20的人脸。<br>
 >>  脚本的位置在：
 ![script](https://github.com/lippman1125/github_images/blob/master/faceboxes_images/wider_2_voc_script.jpg)
 
 >>(2) 在wider_face_2_voc.py的同一级目录中创建wider_face文件夹，放解压好下载的wider数据，如图：<br>
![data](https://github.com/lippman1125/github_images/blob/master/faceboxes_images/wider_2_voc.jpg)
 
 >>(3) 运行wider_face_2_voc.py脚本，在wider_face文件夹中会生成VOC格式的数据，如图：<br>
![data](https://github.com/lippman1125/github_images/blob/master/faceboxes_images/wider_2_voc_data.jpg)

 >>(4_old_method) 生成的图片会把小于20x20的人脸用图像均值覆盖掉，因为太小的人脸，训练时不容易收敛，如图：<br>
![mask](https://github.com/lippman1125/github_images/blob/master/faceboxes_images/wider_small_face_mask.jpg)

 >>(4_new_method) 图片不再预处理，遮盖小的人脸，我们在代码中过滤小的人脸，具体参考提交的代码，如图：<br>
![mask](https://github.com/lippman1125/github_images/blob/master/faceboxes_images/img_demo2.jpg)

 >>(4_new_method) 在文件src/caffe/util/im_transforms.cpp，UpdateBBoxByResizePolicy函数中过滤pixel小于20的人脸，如图：<br>
![mask](https://github.com/lippman1125/github_images/blob/master/faceboxes_images/filter_small_box.jpg)

 >>(5) 利用data/FACE文件中的脚本，把VOC格式转换成LMDB格式，如图：<br>
![lmdb](https://github.com/lippman1125/github_images/blob/master/faceboxes_images/wider_voc_2_lmdb.jpg)

>>在caffe/data目录下创建faces_database文件夹，拷贝wider_face文件夹(前面生成的VOC格式数据)，layout如图：<br>
![database](https://github.com/lippman1125/github_images/blob/master/faceboxes_images/faces_database.bmp)

>>
```C
cd caffe
\# Create the trainval.txt, test.txt, and test_name_size.txt in data/FACE/                
./data/FACE/create_list.sh                                                                
\# You can modify the parameters in create_data.sh if needed.                             
\# It will create lmdb files for trainval and test with encoded original image:           
\#   data/faces_database/FACE/lmdb/FACE_trainval_lmdb                                     
\#   data/faces_database/FACE/lmdb/FACE_test_lmdb                                         
\# and make soft links at examples/FACE/                                                  
./data/FACE/create_data.sh                                                                
```
3. 训练：<br>
---
>> 训练需要的参数文件和网络文件位置如图：<br>
>>![train](https://github.com/lippman1125/github_images/blob/master/faceboxes_images/faceboxes_train.jpg)

>> Paper中的数据预处理，如图：<br>
>> ![paper](https://github.com/lippman1125/github_images/blob/master/faceboxes_images/paper_data_preprocess.jpg)

>> 根据paper我们对train.prototxt的数据采样部门进行了修改，主要是aspect_ratio强制为1，因为是对人脸进行训练，所以没变要改变宽高比, 如图：<br>
>> ![old_sampler](https://github.com/lippman1125/github_images/blob/master/faceboxes_images/old_sampler.jpg)
>> ![new_sampler](https://github.com/lippman1125/github_images/blob/master/faceboxes_images/new_sampler.jpg)

>> 运行以下命令开始训练：<br>
>> ./build/tools/caffe train --solver examples/faceboxes/solver_new.prototxt


4. 测评：<br>
---
>> 测评脚本以及模型文件位置如图：<br>
![test](https://github.com/lippman1125/github_images/blob/master/faceboxes_images/faceboxes_demo.jpg)

>> Paper中的后处理如图：<br>
![bbox_postprocess](https://github.com/lippman1125/github_images/blob/master/faceboxes_images/paper_box_postprocess.jpg)

>> 我们同样根据paper跟新了faceboxes_deploy_new.prototxt，如图：<br>
![deploy_new](https://github.com/lippman1125/github_images/blob/master/faceboxes_images/bbox_postprocess.jpg)

>> FDDB上的老的测评结果(discontinuous)如图：<br>
![old_result](https://github.com/lippman1125/github_images/blob/master/faceboxes_images/faceboxes_roc_train.jpg)

>> FDDB上的新的测评结果(discontinuous)如图：<br>
![new_result](https://github.com/lippman1125/github_images/blob/master/faceboxes_images/fddb_compare2_DiscROC.png)

>> 在同样100 false positives，召回率提升了3个点。

>> 论文中的结果：<br>
![origin](https://github.com/lippman1125/github_images/blob/master/faceboxes_images/faceboxes_roc_origin.jpg)

>> 效果图：<br>
![demo](https://github.com/lippman1125/github_images/blob/master/faceboxes_images/img_demo.jpg)

5. 参考：<br>
---
>>参考的仓库：https://github.com/lsy17096535/faceboxes

6. 优化：<br>
---
>>train.prototxt支持Anchor densification strategy
