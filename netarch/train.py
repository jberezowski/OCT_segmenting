
from tensorflow.keras import optimizers, callbacks

def train_model(data, labels, model, loss, **kwargs):
    model.compile(
        optimizer = optimizers.Adam(lr = 1e-4),
        loss = loss,
        metrics = ['accuracy'],
    )

    model.fit(data, labels, **kwargs)