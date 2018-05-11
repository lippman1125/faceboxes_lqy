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

 >>(4) 生成的图片会把小于20x20的人脸用图像均值覆盖掉，因为太小的人脸，训练时不容易收敛，如图：<br>
![mask](https://github.com/lippman1125/github_images/blob/master/faceboxes_images/wider_small_face_mask.jpg)

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
>> 训练需要的参数文件和网络文件位置如图：
>>![train](https://github.com/lippman1125/github_images/blob/master/faceboxes_images/faceboxes_train.jpg)

>> 运行以下命令开始训练：<br>
>> ./build/tools/caffe train --solver examples/faceboxes/solver.prototxt

4. 测评：<br>
---
>> 测评脚本以及模型文件位置如图：
![test](https://github.com/lippman1125/github_images/blob/master/faceboxes_images/faceboxes_demo.jpg)

>> FDDB上的测评结果(discontinuous)如图：<br>
![result](https://github.com/lippman1125/github_images/blob/master/faceboxes_images/faceboxes_roc_train.jpg)

>> 论文中的结果：<br>
![origin](https://github.com/lippman1125/github_images/blob/master/faceboxes_images/faceboxes_roc_origin.jpg)

>> 效果图：<br>
![demo](https://github.com/lippman1125/github_images/blob/master/faceboxes_images/img_demo.jpg)
