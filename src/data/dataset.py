
import tensorflow as tf
from src.utils.utility import tf_parse



def tf_dataset(X, Y, batch, H ,W):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(lambda x,y: tf_parse(x, y, H, W))
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset