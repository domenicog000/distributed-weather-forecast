from data_pipeline import window_dataset, get_parameters
from model_utility import create_uncompiled_model
from pymongo import MongoClient
import tensorflow as tf
import pandas as pd
import json
import os

def get_measurements():
    client = MongoClient("db", 27017)
    db_name = "weather_data"
    database = client[db_name]
    collection = database["measurements"]
    data = list(collection.find({}, {"_id":0, "Timestamp": 0}))
    return pd.DataFrame(data)

def get_dataset(global_batch_size, total_window_size, shift):
  
  df = get_measurements()

  wv = df['Wind_Speed']
  bad_wv = wv == -9999.0
  df['Wind_Speed'][bad_wv] = 0.0

  n = len(df)
  train_df = df[0: int(n*0.8)]
  val_df = df[int(n*0.8):]

  num_features = df.shape[1]

  train_mean = train_df.mean()
  train_std = train_df.std()

  train_df = (train_df - train_mean) / train_std
  val_df = (val_df - train_mean) / train_std

  len_train = len(train_df)
  len_val = len(val_df)

  train_df = window_dataset(train_df, total_window_size, shift, global_batch_size)
  val_df = window_dataset(val_df, total_window_size, shift, global_batch_size)

  return train_df, len_train, val_df, len_val, num_features

def _is_chief(task_type, task_id):
  return (task_type == 'worker' and task_id == 0) or task_type is None

def _get_temp_dir(dirpath, task_id):
  base_dirpath = 'workertemp_' + str(task_id)
  temp_dir = os.path.join(dirpath, base_dirpath)
  tf.io.gfile.makedirs(temp_dir)
  return temp_dir

def write_filepath(filepath, task_type, task_id):
  dirpath = os.path.dirname(filepath)
  base = os.path.basename(filepath)
  if not _is_chief(task_type, task_id):
    dirpath = _get_temp_dir(dirpath, task_id)
  return os.path.join(dirpath, base)

if __name__ == '__main__':

  tf_config = json.loads(os.environ['TF_CONFIG'])
  
  strategy = tf.distribute.MultiWorkerMirroredStrategy()
  num_workers = strategy.num_replicas_in_sync
  
  per_worker_batch_size = int(os.environ["PER_WORKER_BATCH_SIZE"])
  base_window = int(os.environ["BASE_WINDOW_SIZE"])
  shift = int(os.environ["SHIFT"])
  total_window_size = base_window + shift
  
  global_batch_size = per_worker_batch_size * num_workers

  mse = tf.keras.metrics.MSE
  mae = tf.keras.metrics.MAE
  
  metrics = ["accuracy", mse, mae]
  
  optimizer = tf.keras.optimizers.SGD(learning_rate = 10e-2, momentum=0.9)
  print("Loading Dataset...")
  train, len_train, val, len_val, num_features = get_dataset(global_batch_size, total_window_size, shift)
  
  dataset_size = len_train
  epochs, steps_per_epoch, num_batches = get_parameters(dataset_size, total_window_size, shift, per_worker_batch_size, num_workers)
  validation_steps = num_batches // per_worker_batch_size

  callbacks = []
  worker_id = int(tf_config["task"]["index"])
  if worker_id == 0:
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='.', histogram_freq = 1)
    callbacks.append(tensorboard)
  
  with strategy.scope():
      multi_worker_model = create_uncompiled_model(base_window, shift, num_features=3)
      multi_worker_model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=metrics)
  

  history = multi_worker_model.fit(train, epochs=epochs, validation_data=val, 
                                  steps_per_epoch=steps_per_epoch,
                                  validation_steps=validation_steps,
                                  callbacks=callbacks)
  
  model_path = '/data/model/1/'
  task_type, task_id = (strategy.cluster_resolver.task_type, strategy.cluster_resolver.task_id)
  write_model_path = write_filepath(model_path, task_type, task_id)

  multi_worker_model.save(write_model_path)
