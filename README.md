# GoogleBangkit-MaskClassification

Convolutional Neural Network based Image Classification for Google Bangkit's Final Project Assignment </br>
This project aims to classify images of people wearing and not wearing mask using a TF-based CNN model.

## Requirements
- Python 3.X
- TensorFlow 2.X
- [The dataset](https://www.kaggle.com/ahmetfurkandemr/mask-datasets-v1/kernels) 

## Dataset 
Dataset taken from [Mask Dataset V1](https://www.kaggle.com/ahmetfurkandemr/mask-datasets-v1/kernels) with total of 1100 images. <br />
The dataset is image dataset of people in close-up photo with different poses, background, race, gender, and age. <br />

Train-Validation set contains two labels: <br />
- Mask
- No Mask

The dataset is pre-conditioned (already separated into folders based on its class and into training set and validation set), structured like this: <br />
- Train (750 images)
  - Mask (350 images)
  - No_mask (400 images)
- Validation (350 images)
  - Mask (150 images)
  - No_mask (200 images)

## Baseline Model
[Mask_NoMask_Classification.ipynb](https://github.com/tiadwi/GoogleBangkit-MaskClassification/blob/master/Mask_NoMask_Classification.ipynb) is the baseline model. <br />
For the baseline model, we use Convolutional Neural Network with combination of convolutional, pooling, and dense layers, specified as follows: <br />
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(512, activation='relu'), 
    tf.keras.layers.Dense(1, activation='sigmoid')  
])
```

We also use the `ImageDataGenerator` function from `tensorflow.keras.preprocessing.image` module to augment the training set. The augmentation that we apply to the training set are: rotation, shear, zoom, flip, and width & height shift.

### Result of the baseline model
When training the baseline model, we get the graph of the metrics as follows: <br />
![Baseline model's accuracy](https://github.com/tiadwi/GoogleBangkit-MaskClassification/blob/master/docs/img/1A.png) ![Baseline model's loss](https://github.com/tiadwi/GoogleBangkit-MaskClassification/blob/master/docs/img/1B.png)<br />
Even though the metrics value is good, the validation accuracy is higher than the training accuracy. This is probably because of  "easier" images in the validation set than that of the training set ([source](https://www.researchgate.net/post/When_can_Validation_Accuracy_be_greater_than_Training_Accuracy_for_Deep_Learning_Models)).

The prediction performance is also bad:<br />
<p align="center">
  <img width="200" height="300" src="https://github.com/tiadwi/GoogleBangkit-MaskClassification/blob/master/docs/img/2.png">
</p>
These two problems indicate that our model is highly overfitting. Thus, we need a better alternative model and some tweaks to the dataset.

## Improved Model
[mobilevnet_maskprediction_model.ipynb](https://github.com/tiadwi/GoogleBangkit-MaskClassification/blob/master/mobilevnet_maskprediction_model.ipynb) is the improved model. <br />
To overcome the "easier images" problem in the validation set, we re-shuffle the images to anticipate bias in the pre-conditioned dataset. <br />
We do the re-shuffling procedures as follows:
1. Mix all the Mask images from the pre-defined training and validation folder. Same procedure is applied to the No Mask images.
2. Shuffle both Mask images and No Mask images.
3. Re-distribute the shuffled Mask and No Mask images to training and validation folder.

To improve the prediction performance, we adopt the pre-trained model from [MobileNetV2](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet) to our model. More on the transfer learning using TensorFlow can be accessed through [here](https://www.tensorflow.org/tutorials/images/transfer_learning).

### Result of the improved model
Here are the graph of the metrics and the prediction performance of the improved model:<br />
![Improved model's accuracy](https://github.com/tiadwi/GoogleBangkit-MaskClassification/blob/master/docs/img/3A.png) ![Improved model's loss](https://github.com/tiadwi/GoogleBangkit-MaskClassification/blob/master/docs/img/3B.png)<br />
<p align="center">
  <img width="200" height="300" src="https://github.com/tiadwi/GoogleBangkit-MaskClassification/blob/master/docs/img/4.png">
</p>


## Deployment
We serve the model as an [Android app](https://github.com/tiadwi/GoogleBangkit-MaskClassification/blob/master/MaskClassificationApp.apk) to make it easy to utilize and available on mobile device. The TFLite apps that we used as a template can be obtained through [here](https://github.com/esafirm/bangkit-image-classifier-example).

To deploy the model, first the model is [converted to TFLite using Python API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/convert/python_api.md). The .tflite file produced from the converting process then placed into the asset folder along with the label in .txt format.

The feature of the Android app:
1. Live Camera object classifier<br />
Activating camera and automatically taking picture if it detect a human face, then classify whether the person is wearing a mask or not.<br />
Examples:<br />
![Live Camera example 1](https://github.com/tiadwi/GoogleBangkit-MaskClassification/blob/master/docs/img/5.jpg)![Live Camera example 2](https://github.com/tiadwi/GoogleBangkit-MaskClassification/blob/master/docs/img/6.jpg)
2. Static Image object classifier
Open the gallery to select image (of human face), then classify whether the person is wearing a mask or not.<br />
Examples:<br />
![Static Image example 1](https://github.com/tiadwi/GoogleBangkit-MaskClassification/blob/master/docs/img/7.jpg)![Static Image example 2](https://github.com/tiadwi/GoogleBangkit-MaskClassification/blob/master/docs/img/8.jpg)

## Credit
Presented in 2020, by: <br />
- [Mochammad Randy Caesario Harsuko](https://github.com/mrch-hub) 
- [Muhamad Naufal Azwar Iftikar](https://github.com/mnaufalazwar)
- [Darien Yoga Adi Prawira](https://github.com/darien-yoga)
- [Tia Dwi Setiani](https://github.com/tiadwi)


