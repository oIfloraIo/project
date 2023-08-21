import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.cnn import mini_XCEPTION

# 데이터셋 경로 및 매개변수 설정
train_data_dir = 'archive/train'
validation_data_dir = 'archive/test'
batch_size = 32
num_classes = 7
input_shape = (48, 48, 1)

# 데이터 생성기 설정
train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(48, 48),
                                                    color_mode='grayscale',
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

validation_datagen = ImageDataGenerator(rescale=1.0/255)
validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                              target_size=(48, 48),
                                                              color_mode='grayscale',
                                                              batch_size=batch_size,
                                                              class_mode='categorical')

# 모델 정의 (mini_XCEPTION 모델 사용)
model = mini_XCEPTION(input_shape, num_classes)

# 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
epochs = 100
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples // batch_size)

# 모델 저장
model.save('Emotion_Model_mini_XCEPTION.keras')