---
title:  "classification"
excerpt: “flower_photos 분류”
header:
  teaser: /assets/images/19_cla/output_32_1.png

categories:
  - DL
tags:
  - python
  - tensorflow
  - classification
---

```python
import tensorflow as tf
```

# data 


```python
data = tf.keras.utils.get_file('flower_photos',
                             'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                              untar=True
                            )
```


```python
data
```




    'C:\\Users\\drogpard\\.keras\\datasets\\flower_photos'



# path


```python
import os
```


```python
os.sep
```




    '\\'




```python
os.listdir(data)
```




    ['daisy', 'dandelion', 'LICENSE.txt', 'roses', 'sunflowers', 'tulips']




```python
import pathlib
```


```python
import glob
```


```python
p = pathlib.Path(data)
```


```python
p
```




    WindowsPath('C:/Users/drogpard/.keras/datasets/flower_photos')




```python
img = list(p.glob('*/*.jpg'))
```


```python
# img # classification 용으로는 direcotry별로 이미지 저장 - 관례상 (편의 기능 사용 가능)
```

# pipeline (holdout)


```python
# 1
training = tf.keras.preprocessing.image_dataset_from_directory(p, 
                                                               subset='training', 
                                                               batch_size=64,
                                                               validation_split=0.3, 
                                                               seed=41) # tf.data.Dataset
validation = tf.keras.preprocessing.image_dataset_from_directory(p, 
                                                                 subset='validation', 
                                                                 validation_split=0.3,
                                                                 seed=41) # tf.data.Dataset
```

    Found 3670 files belonging to 5 classes.
    Using 2569 files for training.
    Found 3670 files belonging to 5 classes.
    Using 1101 files for validation.


# batch


```python
training
```




    <BatchDataset element_spec=(TensorSpec(shape=(None, 256, 256, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None,), dtype=tf.int32, name=None))>



# preprocessing


```python
training_ = training.map(lambda x,y : (x/255,y)).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
validation_ = validation.map(lambda x,y : (x/255,y)).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
```

# model ( > Transfer learning)


```python
input_ = tf.keras.Input((256,256,3))
```


```python
x = tf.keras.layers.Conv2D(64, 3)(input_)
x = tf.keras.layers.BatchNormalization()(input_)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPool2D(3,2)(x)
x = tf.keras.layers.Conv2D(128,3)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Conv2D(256,3)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPool2D(3,2)(x)
#x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.GlobalAvgPool2D()(x)
x = tf.keras.layers.Dense(5)(x)
```


```python
model = tf.keras.Model(input_, x)
```


```python
model.summary()
```

    Model: "model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_1 (InputLayer)        [(None, 256, 256, 3)]     0         
                                                                     
     batch_normalization (BatchN  (None, 256, 256, 3)      12        
     ormalization)                                                   
                                                                     
     re_lu (ReLU)                (None, 256, 256, 3)       0         
                                                                     
     max_pooling2d (MaxPooling2D  (None, 127, 127, 3)      0         
     )                                                               
                                                                     
     conv2d_1 (Conv2D)           (None, 125, 125, 128)     3584      
                                                                     
     batch_normalization_1 (Batc  (None, 125, 125, 128)    512       
     hNormalization)                                                 
                                                                     
     re_lu_1 (ReLU)              (None, 125, 125, 128)     0         
                                                                     
     conv2d_2 (Conv2D)           (None, 123, 123, 256)     295168    
                                                                     
     batch_normalization_2 (Batc  (None, 123, 123, 256)    1024      
     hNormalization)                                                 
                                                                     
     re_lu_2 (ReLU)              (None, 123, 123, 256)     0         
                                                                     
     max_pooling2d_1 (MaxPooling  (None, 61, 61, 256)      0         
     2D)                                                             
                                                                     
     global_average_pooling2d (G  (None, 256)              0         
     lobalAveragePooling2D)                                          
                                                                     
     dense (Dense)               (None, 5)                 1285      
                                                                     
    =================================================================
    Total params: 301,585
    Trainable params: 300,811
    Non-trainable params: 774
    _________________________________________________________________



```python
import tensorflow_addons as tfa
```


```python
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['acc'])
```

# train


```python
# callbaack 
```


```python
tf.keras.callbacks.EarlyStopping
tf.keras.callbacks.LearningRateScheduler
tf.keras.callbacks.ReduceLROnPlateau
tf.keras.callbacks.ModelCheckpoint
```




    keras.callbacks.ModelCheckpoint




```python
history = model.fit(training_, epochs=3, validation_data=validation_)
```

    Epoch 1/3
    41/41 [==============================] - 692s 17s/step - loss: 1.2940 - acc: 0.4644 - val_loss: 1.6341 - val_acc: 0.2352
    Epoch 2/3
    41/41 [==============================] - 720s 18s/step - loss: 1.1630 - acc: 0.5325 - val_loss: 1.7552 - val_acc: 0.2343
    Epoch 3/3
    41/41 [==============================] - 724s 18s/step - loss: 1.1033 - acc: 0.5613 - val_loss: 1.7961 - val_acc: 0.2343



```python
import pandas as pd
```


```python
pd.DataFrame(history.history).plot.line()
```




    <AxesSubplot:>




    
![png](/assets/images/19_cla/output_32_1.png)
    



```python

```
