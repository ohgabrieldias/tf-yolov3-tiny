import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D
from config import *
from Dataset import YOLODatasetTF
from Loss import yolo_tiny_loss
# Ativa uso eficiente da GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True) 

# === CONFIGURAÇÃO GLOBAL ===
IMAGE_SIZE = 416
CHECKPOINT_DIR = "checkpoints_qat"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# === FUNÇÕES AUXILIARES ===
def export_qat_to_tflite(model, export_path="yolov3_tiny_qat.tflite"):
    # Converter o modelo quantizado (QAT) para TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()

    with open(export_path, "wb") as f:
        f.write(tflite_quant_model)

    print(f"✅ Modelo quantizado salvo como TFLite em: {export_path}")

# === MODELO SIMPLIFICADO ===
def conv_bn_leaky(x, filters, kernel_size, strides=1):
    padding = 'same' if strides == 1 else 'valid'
    if strides > 1:
        x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

def yolov3_tiny(input_shape=(416, 416, 3), num_classes=1):
    inputs = Input(shape=input_shape)

    x = conv_bn_leaky(inputs, 16, 3)
    x = MaxPooling2D(2, 2, padding='same')(x)

    x = conv_bn_leaky(x, 32, 3)
    x = MaxPooling2D(2, 2, padding='same')(x)

    x = conv_bn_leaky(x, 64, 3)
    x = MaxPooling2D(2, 2, padding='same')(x)

    x = conv_bn_leaky(x, 128, 3)
    x = MaxPooling2D(2, 2, padding='same')(x)

    x = conv_bn_leaky(x, 256, 3)
    x = MaxPooling2D(2, 1, padding='same')(x)

    x = conv_bn_leaky(x, 512, 3)
    x = MaxPooling2D(2, 1, padding='same')(x)

    x = conv_bn_leaky(x, 1024, 3)
    x = conv_bn_leaky(x, 256, 1)

    output_filters = 3 * (5 + num_classes)
    detection = Conv2D(output_filters, 1, 1, padding='same', name='detection_layer')(x)

    return Model(inputs, detection, name='yolov3_tiny_simplified')

# === EXEMPLO DE USO ===
if __name__ == "__main__":
    # Carregar dataset (ajuste o caminho conforme seu projeto)
    dataset = YOLODatasetTF(root_dir=DATASET_PATH, subset="train").batch(4)

    model = yolov3_tiny()


quantize_model = tfmot.quantization.keras.quantize_model
qat_model = quantize_model(model)
print("✅ Modelo QAT criado!")

# Compile com a loss YOLO
qat_model.compile(optimizer='adam', loss=yolo_tiny_loss)
# ======== Callback para salvar a cada 15 épocas ========

class PeriodicSaver(tf.keras.callbacks.Callback):
    def __init__(self, save_dir, interval=15):
        super().__init__()
        self.save_dir = save_dir
        self.interval = interval
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            filename = f"qat_epoch_{epoch+1:03d}.keras"
            path = os.path.join(self.save_dir, filename)
            self.model.save(path)
            print(f"\n✅ Modelo salvo em {path}")



# ======== Treinamento ========
qat_model.fit(
    dataset,
    epochs=EPOCHS,
    callbacks=[PeriodicSaver(CHECKPOINT_DIR, interval=5)]
)



export_qat_to_tflite(qat_model)
