# Handwritten-CNN-method-minist-dataset-detection



**This is a pure code project of the CNN method to detect the Mnist dataset!**

My work has take the **victorzhou**'s Blog and a bilibili up named "天外来客" as conferences.

## files:

test_conv.py : only a cnn layer test

add_pooling.py : add pooling layer after the cnn layer

add_softmax.py : add a softmax to predict the number(It can run to detect the number without pretrain)

Training.py : utilize gradient descent method and backprop method to train 

All of the above four files can be run by  python  XXX.py

## environment requirements:

tensorflow                2.1.0 

numpy                     1.19.2 

python 					 3.6.13

keras                    	2.3.1

Some explanation has already put in the code files

![1712762888641](https://github.com/crazy1212122/Handwritten-CNN-method-minist-dataset-detection/assets/109590350/fd647595-8a89-44e3-8119-d0552d0817b9)


### add_softmax.py

**An example without pretraining detection** ,which only contains a fully connected network  and detect without pretrain

here is an example

![1712761512496](https://github.com/crazy1212122/Handwritten-CNN-method-minist-dataset-detection/assets/109590350/a53400d9-3221-42a8-9716-3d9518e716b0)


### Training.py

**Utilize** **gradient descent method**  **and backprop method to train**  before detect.

an example of 5000 steps pre-train and then take 1000 samples to test: 

![1712763003131](https://github.com/crazy1212122/Handwritten-CNN-method-minist-dataset-detection/assets/109590350/004fc0d2-1159-4a8d-945d-8d92f6456669)


**Some  detailed math process are put in the docx file**

（I'll translate it into English later...)

