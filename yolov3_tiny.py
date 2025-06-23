import tensorflow as tf
from tensorflow.keras import layers, Model

def conv_block(x, filters, kernel_size, strides=1):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same',
                      use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    return x

def yolov3_tiny(input_shape=(416, 416, 3), num_classes=1):
    inputs = layers.Input(shape=input_shape)

    x = conv_block(inputs, 16, 3)
    x = layers.MaxPooling2D(2, 2, padding='same')(x)

    x = conv_block(x, 32, 3)
    x = layers.MaxPooling2D(2, 2, padding='same')(x)

    x = conv_block(x, 64, 3)
    x = layers.MaxPooling2D(2, 2, padding='same')(x)

    x = conv_block(x, 128, 3)
    x = layers.MaxPooling2D(2, 2, padding='same')(x)

    x = conv_block(x, 256, 3)
    route_1 = x  # Para detecção em 26x26

    x = layers.MaxPooling2D(2, 2, padding='same')(x)

    x = conv_block(x, 512, 3)
    x = layers.MaxPooling2D(2, 1, padding='same')(x)

    x = conv_block(x, 1024, 3)
    x = conv_block(x, 256, 1)  # redução de canais para upsample depois

    # Detecção na escala 13x13
    x1 = conv_block(x, 512, 3)
    pred_small = layers.Conv2D(3 * (5 + num_classes), 1, padding='same')(x1)

    # Detecção na escala 26x26
    x = conv_block(x, 128, 1)
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate()([x, route_1])
    x = conv_block(x, 256, 3)
    pred_medium = layers.Conv2D(3 * (5 + num_classes), 1, padding='same')(x)

    model = Model(inputs, [pred_small, pred_medium])
    return model