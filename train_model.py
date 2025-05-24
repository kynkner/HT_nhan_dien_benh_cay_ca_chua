import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import os

# Cấu hình cơ bản
train_dir = "dataset/train"
# val_dir = "dataset/validation"
image_size = (224, 224)
batch_size = 32
num_classes = 10

# 1. ImageDataGenerator 
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# validation_generator = val_datagen.flow_from_directory(
#     val_dir,
#     target_size=image_size,
#     batch_size=batch_size,
#     class_mode='categorical',
#     shuffle=False
# )

# 2. Mô hình Transfer Learning với MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# 3. Đóng băng layers pre-trained
for layer in base_model.layers:
    layer.trainable = False

# 4. Compile model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Train model
model.fit(
    train_generator,
    epochs=30,
    # validation_data = validation_generator,
)

# 6. Lưu model
model.save("model/sweetpotato_disease_model.h5")
