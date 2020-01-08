import tensorflow as tf
from tensorflow.keras import layers, models

def ReNetCell(input_layer, hidden_layer_size=None):
    _, nrows, ncols, nchannels = input_layer.shape
    if hidden_layer_size is None:
        hidden_layer_size
    #batch_size, nrows, ncols, nchannels
    transposed_input = tf.transpose(input_layer, perm=[0, 2, 1, 3])
    #batch_size, ncols, nrows, nchannels
    reshaped_input = tf.reshape(transposed_input, (-1, nrows, nchannels))
    #batch_size*ncols, nrows, nchannels
    
    output =layers.Bidirectional(
        layers.LSTM(nchannels, return_sequences=True),
        input_shape=(nrows, nchannels),
    )(reshaped_input)
    
    reshaped_output = tf.reshape(output, (-1, ncols, nrows, 2*nchannels))
    #batch_size, ncols, nrows, nchannels
    transposed_output = tf.transpose(reshaped_output, perm=[0, 2, 1, 3])
    #batch_size, nrows, ncols, nchannels
    return transposed_output

def LSTM_Unet():

    INPUT_SIZE = (496, 496, 1)

    KERNEL = 3
    NUM_CLASSES = 9
    POOL_SIZE = (2, 2)

    CONV_PARAMS = {
        'activation': 'relu',
        'padding': 'same',
        'kernel_initializer': 'he_normal'
    }

    inputs = layers.Input(INPUT_SIZE)

    #Downward slope
    down_conv_1 = layers.Conv2D(32, KERNEL, **CONV_PARAMS)(inputs)
    down_conv_1 = layers.Conv2D(32, KERNEL, **CONV_PARAMS)(down_conv_1)
    down_pool_1 = layers.MaxPooling2D(POOL_SIZE)(down_conv_1)

    down_conv_2 = layers.Conv2D(64, KERNEL, **CONV_PARAMS)(down_pool_1)
    down_conv_2 = layers.Conv2D(64, KERNEL, **CONV_PARAMS)(down_conv_2)
    down_pool_2 = layers.MaxPooling2D(POOL_SIZE)(down_conv_2)

    down_conv_3 = layers.Conv2D(128, KERNEL, **CONV_PARAMS)(down_pool_2)
    down_conv_3 = layers.Conv2D(128, KERNEL, **CONV_PARAMS)(down_conv_3)
    down_pool_3 = layers.MaxPooling2D(POOL_SIZE)(down_conv_3)

    down_conv_4 = layers.Conv2D(256, KERNEL, **CONV_PARAMS)(down_pool_3)
    down_conv_4 = layers.Conv2D(256, KERNEL, **CONV_PARAMS)(down_conv_4)
    down_pool_4 = layers.MaxPooling2D(POOL_SIZE)(down_conv_4)

    #Re-net Center
    down_pool_last = down_pool_4
    bidirect = ReNetCell(down_pool_last)
    center_drop = bidirect
    #Upward slope

    up_sampling_conv_4 = layers.Conv2D(256, 2, **CONV_PARAMS)(layers.UpSampling2D(size = POOL_SIZE)(center_drop))
    up_merge_4 = layers.Concatenate(axis=3)([up_sampling_conv_4, down_conv_4])
    up_conv_4 = layers.Conv2D(256, KERNEL, **CONV_PARAMS)(up_merge_4)
    up_conv_4 = layers.Conv2D(256, KERNEL, **CONV_PARAMS)(up_conv_4)

    up_sampling_conv_3 = layers.Conv2D(128, 2, **CONV_PARAMS)(layers.UpSampling2D(size = POOL_SIZE)(up_conv_4))
    up_merge_3 = layers.Concatenate(axis=3)([up_sampling_conv_3, down_conv_3])
    up_conv_3 = layers.Conv2D(128, KERNEL, **CONV_PARAMS)(up_merge_3)
    up_conv_3 = layers.Conv2D(128, KERNEL, **CONV_PARAMS)(up_conv_3)

    up_sampling_conv_2 = layers.Conv2D(64, 2, **CONV_PARAMS)(layers.UpSampling2D(size = POOL_SIZE)(up_conv_3))
    up_merge_2 = layers.Concatenate(axis=3)([up_sampling_conv_2, down_conv_2])
    up_conv_2 = layers.Conv2D(64, KERNEL, **CONV_PARAMS)(up_merge_2)
    up_conv_2 = layers.Conv2D(64, KERNEL, **CONV_PARAMS)(up_conv_2)

    up_sampling_conv_1 = layers.Conv2D(32, 2, **CONV_PARAMS)(layers.UpSampling2D(size = POOL_SIZE)(up_conv_2))
    up_merge_1 = layers.Concatenate(axis=3)([up_sampling_conv_1, down_conv_1])
    up_conv_1 = layers.Conv2D(32, KERNEL, **CONV_PARAMS)(up_merge_1)
    up_conv_1 = layers.Conv2D(32, KERNEL, **CONV_PARAMS)(up_conv_1)

    output = layers.Conv2D(
        NUM_CLASSES,
        1,
        activation = None,
        padding='same',
        kernel_initializer='he_normal',
    )(up_conv_1)
    output = layers.Softmax(axis=-1)(output)

    model = models.Model(inputs = inputs, outputs = output)

    return model
