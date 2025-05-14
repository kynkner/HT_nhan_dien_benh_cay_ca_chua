from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
import os

# Tạo thư mục cho mô hình nếu chưa có
os.makedirs("model", exist_ok=True)

# Tải MobileNetV2 
base_model = MobileNetV2(include_top=False, weights='imagenet', input_tensor=Input(shape=(224, 224, 3)))
base_model.trainable = False

# Thêm các lớp phân loại cho 10 lớp
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(10, activation='softmax')(x)  # 10 lớp: từ healthy đến các bệnh

# Tạo mô hình hoàn chỉnh
model = Model(inputs=base_model.input, outputs=x)

# Compile mô hình
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Lưu mô hình
model.save("model/mobilenet_model.h5")
