#import matplotlib.pyplot as plt
import tensorflow as tf

def create_uncompiled_model(window_size, label_size, num_features):
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                            strides=1,
                            activation="relu",
                            padding='causal',
                            input_shape=[window_size, 3]),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dense(label_size * num_features),
        #tf.keras.layers.Lambda(lambda x: x * 400),
        tf.keras.layers.Reshape([label_size, num_features])
        ])

    return model

#def compile_and_fit(model, window, patience = 2, learning_rate=None):
#    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = patience, mode = 'min')
#    model.compile(loss = tf.keras.losses.Huber(), optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9), metrics = [tf.keras.metrics.MeanAbsoluteError()]) 
#    #history = model.fit(window.train, epochs = EPOCHS, validation_data = window.val, callbacks = [early_stopping])
#    history = model.fit(window.train, epochs = EPOCHS, validation_data = window.val)
#    return history

def compile_and_fit(model, optimizer, loss, train_data, epochs, callbacks, metrics, val_data = None):
    #model = create_uncompiled_model(window_size, label_size, num_features)
    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
    history = model.fit(train_data, validation_data = val_data,  epochs = epochs, callbacks = callbacks)
    return history

'''def plot_learning_rate(x, loss, boundaries):
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.semilogx(x, loss)
    plt.tick_params('both', length=10, width=1, which='both')
    plt.axis(boundaries)
    plt.show()'''