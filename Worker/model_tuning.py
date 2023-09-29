from model_utility import compile_and_fit
import tensorflow as tf
import pandas as pd
import numpy as np

EPOCHS = 20

def tuning(window, total_window_size, shift, num_features):
    # Model Tuning
    model_tune = create_uncompiled_model(total_window_size, shift, num_features)
    
    optimizer_tune = tf.keras.optimizers.SGD(momentum=0.9)
    loss_tune = tf.losses.Huber()
    
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (epoch / 20))
    callbacks_tune = [lr_schedule]
    
    mse = tf.keras.metrics.MSE
    mae = tf.keras.metrics.MAE
    #rsqr = tfa.metrics.RSquare()
    
    metrics = [mse, mae]
    
    history_tune = compile_and_fit(model_tune, optimizer_tune, loss_tune, window.train,
                                   EPOCHS, callbacks_tune, metrics = metrics)
    
    x = 1e-3 * (10 ** (np.arange(EPOCHS) / 20))
    loss = history_tune.history['loss']
    boundaries = [1e-3 * (10 ** (1 / 20)), 1e-3 * (10 ** (100 / 20)), -0.1, 0.6]
    
    #plot_learning_rate(x, loss, boundaries)
    
def create_uncompiled_model(window_size, label_size, num_features):
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                            strides=1,
                            activation="relu",
                            padding='causal',
                            input_shape=[window_size, num_features]),  
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dense(label_size * num_features),
        tf.keras.layers.Reshape([label_size, num_features])
        ])

    return model

def windowed_dataset(series, total_window_size, shift, batch_size, shuffle_buffer=1000):
    
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(total_window_size, shift=shift, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(total_window_size))
    dataset = dataset.map(lambda window: (window[:-shift], window[-shift:]))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size).prefetch(1)

    return dataset

def get_dataset(global_batch_size=16):

    df = pd.read_csv('data.csv')

    timestamp = df['Timestamp'][5::6]
    pressure = df['Pressure'][5::6]
    temperature = df['Temperature'][5::6]
    wind_speed = df['Wind_Speed'][5::6]

    df.pop('Timestamp')

    wv = df['Wind_Speed']
    bad_wv = wv == -9999.0
    wind_speed[bad_wv] = 0.0

    n = len(df)
    train_df = df[0: int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]
        
    num_features = df.shape[1]

    # Data Normalization
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std
    
    train_df = windowed_dataset(train_df, 288, 144, global_batch_size)
    val_df = windowed_dataset(val_df, 288, 144, global_batch_size)
    test_df = windowed_dataset(test_df, 288, 144, global_batch_size)

    return train_df, val_df, test_df

def get_epochs(dataset_size, window_size, shift, batch_size, num_workers):
    num_windows = (dataset_size - window_size) // shift + 1
    global_batch_size = batch_size * num_workers
    num_batches = num_windows // global_batch_size + 1
    step_per_epoch = num_batches // batch_size
    epochs = num_batches // step_per_epoch
    return epochs, step_per_epoch

if __name__ == '__main__':

    dataset_size = 293928
    window_size = 288
    label = 144
    batch_size = 16
    num_workers = 2
    
    epochs, step_per_epoch = get_epochs(dataset_size, window_size, label, batch_size, num_workers)
    print(epochs, " ", step_per_epoch)
    model = create_uncompiled_model(144, 144, 3)
    
    optimizer = tf.keras.optimizers.SGD(learning_rate = 10e-2, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(), optimizer = optimizer)
    
    train, val, test = get_dataset()
    
    model.fit(train, epochs = epochs, validation_data = val, steps_per_epoch=step_per_epoch)
    
    