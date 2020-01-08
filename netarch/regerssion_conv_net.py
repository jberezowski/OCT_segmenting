from tensorflow.keras import layers, models


def Regression_Conv_Unet():
    INPUT_SIZE = (496, 496, 1)

    KERNEL = 3
    NUM_CLASSES = 1
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

    #U-net Center
    down_pool_last = down_pool_4
    center_conv = layers.Conv2D(512, KERNEL, **CONV_PARAMS)(down_pool_last)
    center_conv = layers.Conv2D(512, KERNEL, **CONV_PARAMS)(center_conv)
    center_drop = layers.Dropout(0.5)(center_conv)

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
        activation='sigmoid',
        padding='same',
        kernel_initializer='he_normal',
    )(up_conv_1)

    model = models.Model(inputs = inputs, outputs = output)

    return model