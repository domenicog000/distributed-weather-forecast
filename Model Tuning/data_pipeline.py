import tensorflow as tf

def window_dataset(series, total_window_size, shift, batch_size, shuffle_buffer=1000):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(total_window_size, shift=shift, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(total_window_size))
    dataset = dataset.map(lambda window: (window[:-shift], window[-shift:]))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size).prefetch(1)
    
    return dataset

def get_parameters(dataset_size, window_size, shift, batch_size, num_workers):
    num_windows = (dataset_size - window_size) // shift + 1
    global_batch_size = batch_size * num_workers
    num_batches = num_windows // global_batch_size + 1
    steps_per_epoch = num_batches // batch_size
    epochs = num_batches // steps_per_epoch
    return epochs, steps_per_epoch, num_batches

if __name__ == '__main__':

    dataset = tf.data.Dataset.from_tensor_slices(tf.range(240))
    dataset = dataset.window(24, shift=24, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(24))
    dataset = dataset.map(lambda window: (window[:-1], window[-1:]))

    for X, y in dataset:
        print(X.numpy(), " ", y.numpy())