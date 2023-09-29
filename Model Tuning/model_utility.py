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
        tf.keras.layers.Reshape([label_size, num_features])
        ])

    return model
