import tensorflow as tf
from tensorflow.keras import Model, layers

def DarknetConv(x, filters, kernel_size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        x = layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
        padding = 'valid'
    
    x = layers.Conv2D(filters, kernel_size, strides=strides, 
                      padding=padding, use_bias=not batch_norm)(x)
    
    if batch_norm:
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
    return x

def DarknetBlock(x, filters):
    shortcut = x
    x = DarknetConv(x, filters, 1)
    x = DarknetConv(x, filters*2, 3)
    x = layers.Add()([shortcut, x])
    return x

def YOLOv3Tiny(input_shape=(416, 416, 3), num_classes=1):
    inputs = layers.Input(input_shape)
    
    # Backbone
    x = DarknetConv(inputs, 16, 3)
    x = layers.MaxPooling2D(2, 2, 'same')(x)
    x = DarknetConv(x, 32, 3)
    x = layers.MaxPooling2D(2, 2, 'same')(x)
    x = DarknetConv(x, 64, 3)
    x = layers.MaxPooling2D(2, 2, 'same')(x)
    x = DarknetConv(x, 128, 3)
    x = layers.MaxPooling2D(2, 2, 'same')(x)
    x = DarknetConv(x, 256, 3)
    
    # First detection branch
    route1 = x
    x = layers.MaxPooling2D(2, 2, 'same')(x)
    x = DarknetConv(x, 512, 3)
    x = layers.MaxPooling2D(2, 1, 'same')(x)
    x = DarknetConv(x, 1024, 3)
    
    # Output 1 (13x13 for 416 input)
    output1 = DarknetConv(x, 256, 1)
    output1 = DarknetConv(output1, 512, 3)
    output1 = DarknetConv(output1, 3*(num_classes + 5), 1, batch_norm=False)
    output1 = layers.Reshape((13, 13, 3, num_classes + 5))(output1)
    
    # Second detection branch
    x = DarknetConv(route1, 128, 1)
    x = layers.UpSampling2D(2)(x)
    
    # Output 2 (26x26 for 416 input)
    output2 = layers.Concatenate()([x, route1])
    output2 = DarknetConv(output2, 256, 3)
    output2 = DarknetConv(output2, 3*(num_classes + 5), 1, batch_norm=False)
    output2 = layers.Reshape((26, 26, 3, num_classes + 5))(output2)
    
    return Model(inputs, [output1, output2])