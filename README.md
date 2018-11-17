## 7. Semantic Segmentation

### Self-Driving Car Engineer Nanodegree Program - Term 3

### Goal
This project is to recognize and semantically segment the roads for self driving cars by labeling the pixels of a road in images using a Fully Convolutional Network (FCN).

### Demo

The final program is implemented in Tensorflow using a Fully Convolutional Network (FCN). It has successfully labeled the pixels of roads on the testing images. Example results are in the following. 

[//]: # (Image References)
[image1]: ./images/animation.gif "animated result"
[image2]: ./images/umm_000046.png "example result"
[image3]: ./images/fnc_schema.png "FCN model"

![alt text][image1] <br/>
![alt text][image2] <br/>

### Essentials

Essential steps of the implementaion are in a few functions marked with TODOs in the `main.py` file. 

#### load_vgg function

Tensorflow `saved_model.loader` loads a pre-trained VGG16 model. This is the first half of the FCN. The model graph as welll as the pre-trained weights will be reused and retrained like what we did in transfer learning, except that the outputs of certain intermediate layers are also used. Here we extract input, keep, layer3, layer4, and layer7.  

#### layers function

The layers function implements the second half of the FCN. The outputs of VGG layer7, layer4, and layer3 are wired the same way as the schema from the FCN paper in the following.

![alt text][image2] <br/>

VGG layer7, layer4, and layer3 outputs are wired to 4x4 and 1x1 convolution layers, and upsampled to a 16x16 convolution layer. Two skip layers are also added to recover some fine-grained spatial information.

The downsampling path is used to extract and interpret the context (what), while the upsampling path is used to enable precise localization (where). 

It is important to have a kernel initializer and kernel regularizer implemented in each layers to keep the training converge. 

````
    kernel_initializer = tf.random_normal_initializer(stddev=1e-2),
    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
````

#### optimize function

An Adam optimizer has been used to minimize the crosse entropy loss. 

#### train_nn function

Input images and correct labels are fed into the FCN model with tensorflow session runs, as in typical tensorflow training. 

#### run function

Prepares the input place holders and call the functions implemented. 

#### hyperparameters

Several hyperparameters have been experimented. The final set that works for the results are in the following:

````
    epochs 46
    batch_size 5
    learning rate: 0.0001
    keep_prob: 0.5
````

### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip).
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [post](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/Semantic_Segmentation_advice.pdf) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
