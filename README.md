
### Identifying Curb Ramps on Sidewalks through Google Street View images

When I moved to Seattle, one year ago, I noticed that some neighborhoods were not served with disabled-friendly sidewalks, especially because not every corner had a curb ramp. Fortunatelly, the City is making an effort for adding [ADA (Americans with Disabilities Act) compliant curb ramps](http://www.ada-compliance.com/ada-compliance/cub-ramp) on its streets. 

<p align="center"> <img src="/images/ADA.png" width="90%"></p>

The Seattle Department of Transportation (SDOT) just started implementing a [Pedestrian Master Plan](https://www.theurbanist.org/2018/02/08/sdot-unveils-first-five-year-pedestrian-implementation-plan/), that required a minutious assessment of curb ramp’s condition. However this work is costly, usually made by mapathons, and not all cities have enough resources. 

So, the goal of my project is to try to do this task by recognizing the curb ramps directly on Google Street View images, using convolutional neural network. More specifically, Tensorflow Object Detection. 
  
### 1. Extracting images from Google Street View

The images used for this project were the pictures of intersections, extracted by Google Street View API. Fortunately, a University of Washington project named [AccessMap](https://accessmap.io), designed to improve the sidewalk data for pedestrians, provided the coordinates of all intersections in Seattle, using SDOT data.

After that, I manually labelled 1500 images, drawing rectangles around curb ramps for teaching the model what object I’m trying to recognize (I will explain it later).

<p align="center"> <img src="/images/data_collection.png" width="90%"></p>

### 2. Choosing the Model

#### Method for identifying curb ramps on images

There are different ways of identifying a  specific class on images, as illustrated in the picture below.

<p align="center"> <img src="/images/image_detection.jpg" width="90%"></p>
<p align="center"><font size="1">Source: <a href="https://research.fb.com/learning-to-segment/">Facebook Research</a></font></p> 

For this project I chose the object detection method. First, because since I'm training images with custom classes, it requires drawing simple bounding boxes around the ramps, instead of polygons as in segmentation. And I believe it would be a tricky task to use the classification method on a new category. 

Second, as a path to my quest, the Tensorflow Object Detection API - realeased by Google in 2017 - facilitates the task.

#### Tensorflow Object Detection API:

The Object Detection API has been trained on Microsoft COCO dataset (a dataset of about 300,000 images of 90 commonly found objects) with different trainable detection models. The API does the fine-tuning on a pre-trained object detection model with a custom data set and new classes (process called [transfer learning](https://www.tensorflow.org/tutorials/image_retraining)), removing the last 90 neuron classification layer of the network and replacing it with a new layer that outputs 2 categories ("yellow curb ramp" and "gray curb ramp"). It also includes image augmentation, such as flipping and saturation.

#### Transfer Learning Architecture

With the transfer learning process is possible to shorten the amount of time needed to train the entire model, by taking adjantage of an existing model and retrain its final layer(s) to detect curb ramps for us. 

Tensorflow does offer a few models (in the tensorflow [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models-coco-models)).

There's a speed/accuracy trade-off when choosing the object detection model, as despicted in the image below:

<p align="center"> <img src="/images/models_trade-off.jpg" width="65%"></p>
<p align="center"><font size="1">Source: <a href="https://arxiv.org/pdf/1611.10012.pdf">Speed/accuracy trade-offs for modern convolutional object detectors</a></font></p>
  
The sweet spot is the “elbow” part of the mAP (Mean Average Precision) vs GPU time graph. Based on that, I chose to use [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf) object detection model, with [RestNet](https://arxiv.org/abs/1512.03385) feature extractor, trained on [COCO](http://cocodataset.org) dataset.

<p align="center"> <img src="/images/transferlearning.png" width="80%"></p>

### 3. Preparing the Data

#### Label the images

First, I filtered the streets' intersections images that were classified by SDOT as having curb ramps. Afterwards, I draw retangles around yellow (with tactile warning) and grey curb ramps (without tactile warning) in 1500 images using [VOTT](https://github.com/Microsoft/VoTT/releases). I found this labelling tool more user-friendly than Rectlabel.

#### Convert data to TFRecord format

Tensorflow Object Detection API uses the TFRecord file format, so at the end we need to convert our dataset to this file format. I generated a tfrecord using a code adapted from this [raccoon detector](https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py) .


### 4. Training the Model

Once I decided the architecture, the first step for training, was to download the Faster-RCNN_RestNet model.

```
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz
tar xvzf faster_rcnn_resnet50_coco_2018_01_28.tar.gz
```
For installing the Object Detection API, you need to run the following code on the root directory:

```
git clone https://github.com/tensorflow/models.git
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd ..
cd ..
```
Finally, I could train the model, using the command:

 ```
 python3 models/research/object_detection/train.py \
 --logtostderr \
 --train_dir=${PATH_TO_TRAIN_DIR} \
 --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG}
 ```

I used tensorflow-gpu 1.5 on a Win10 machine with a NVIDIA GeForce GTX 970 4GB, following the installation steps described on the [Tensorflow website](https://www.tensorflow.org/install/install_windows). By using a GPU, the training was 10+ times faster than using tensorflow without GPU support on a MacBook.

Once training was complete it was time to test the model. The following command export the inference graph based on the best checkpoint:

```
python3 models/research/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix ${TRAIN_PATH} \
    --output_directory object_detection_graph
```

### 5. Results!

I tested a few pictures to check if it identifies the curb ramps. I was very happy with the results so far. The results using Faster R-CNN RestNet  were significantly more accurate than in my previous attempt of using SSD MobileNet. You can check some predictions below.

<p align="center"> <img src="/images/results.png" width="80%"></p>

With 75% recall (% of ramps that could be detected) and 80% precision (% correctly detected), the model identified curb ramps, classifying them as likely to be ADA compliant or not compliant. The majority of wrong predictions are false negatives, usually because the ramp is not entirely on the picture, or is distant or all covered by shadow.

### 6. Next Steps

* Adjust image augmentation to improve the precision.
* Improve the algorithm for extracting images of corners from Google Street View, to make sure the ramp is entirely on the image.
* Expand beyond Fremont.
