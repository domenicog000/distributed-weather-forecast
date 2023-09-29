import tensorflow as tf

def window_dataset(series, total_window_size, shift, batch_size, shuffle_buffer=1000):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(total_window_size, shift=shift, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(total_window_size))
    dataset = dataset.map(lambda window: (window[:-shift], window[-shift:]))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size).prefetch(1)
    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.with_options(options)

    return dataset

def get_parameters(dataset_size, window_size, shift, batch_size, num_workers):
    num_windows = (dataset_size - window_size) // shift + 1
    global_batch_size = batch_size * num_workers
    num_batches = num_windows // global_batch_size + 1
    steps_per_epoch = num_batches // batch_size
    epochs = num_batches // steps_per_epoch
    return epochs, steps_per_epoch, num_batches
    