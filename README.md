# Keras Mobile Colorizer

Utilizes a U-Net inspired model conditioned on MobileNet class features to generate a mapping from Grayscale to Color image. 
Based on the work https://github.com/baldassarreFe/deep-koalarization

Uses MobileNets for memory efficiency in comparison to Inception-ResNet-V2 so that training can be done on a single GPU (of 4 GB size minimum).

# Installation 
Open the `data_utils.py` script and edit the `TRAIN_IMAGE_PATH` and `VALIDATION_IMAGE_PATH` to point to directories of images. There must be at least 1 folder pointed to by each of those paths.

Then run `data_utils.py` to construct the required folders and the TFRecords which will store the training data. 

This is necessary to drastically improve the speed of training by extracting all the MobileNet features from each training image before training. The major bottleneck during training is the extraction of image features from MobileNet at runtime.

# Training & Evaluation

- To train the model : Use the `train_mobilenet.py` script. Make sure to verify the batch size and how many images are in the TF record before beginning training.

- To evaluate the model : Use the `evaluate_mobilenet.py` script. Make sure that the path to the validation images is provided in `data_utils.py`

# Evaluation
There are a lot of splotchy reddish-brown patches. This may probably be because training was done using only 60k images from MS-COCO dataset, not the full ImageNet dataset.

<img src="https://github.com/titu1994/keras-mobile-colorizer/blob/master/images/page1.png?raw=true" width=100%>

<img src="https://github.com/titu1994/keras-mobile-colorizer/blob/master/images/page2.png?raw=true" width=100%>

<img src="https://github.com/titu1994/keras-mobile-colorizer/blob/master/images/page3.png?raw=true" width=100%>

# Requirements
- Keras 2.0.8+
- Numpy
- Scikit-image
- Tensorflow (GPU is a must for training, CPU is fine for inference)

Install via `pip install -r "requirements.txt"`
