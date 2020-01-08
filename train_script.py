import sys

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint


from netarch.conv_unet import Conv_Unet
from netarch.lstm_unet import LSTM_Unet
from netarch.regerssion_conv_net import Regression_Conv_Unet

from netarch.loss import bce_dice_loss
from netarch.train import train_model


MODELS = {
    'unet': Conv_Unet,
    'renet': LSTM_Unet,
    'regunet': Regression_Conv_Unet,
}

LOSSES = {
    'mse': 'mse',
    'dice': bce_dice_loss,
}

def main():
    tf.random.set_seed(1)
    np.random.seed(1)

    model_to_use = sys.argv[1]
    epochs = int(sys.argv[2])

    images = np.load('./data/images.npy')
    images = images.reshape(images.shape + (1, ))
    labels = np.load(f'./data/{sys.argv[3]}.npy')

    X_train, X_test, y_train, y_test = train_test_split(
        images.astype(np.float16),
        labels,
        test_size=0.1,
        random_state=42
    )

    model_checkpoint = ModelCheckpoint(
        './saved_weights/' + model_to_use + '/weights-2-{epoch:02d}.hdf5',
        save_best_only=False,
        verbose=1,
        period=5,
    )
    model = MODELS[model_to_use]()
    model.load_weights('./saved_weights/' + model_to_use + '/weights-200.hdf5')
    train_model(
        data=X_train,
        labels=y_train,
        model=model,
        loss=LOSSES[sys.argv[4]],
        epochs=epochs,
        batch_size=3,
        callbacks=[model_checkpoint]
    )

if __name__ == "__main__":
    main()
