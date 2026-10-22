# Functions from AVGN (https://github.com/timsainb/avgn_paper)

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import matplotlib.pyplot as plt


def serialize_example(example):
    """Serialize an item in a dataset
    Arguments:
      example {[list]} -- list of dictionaries with fields "name" , "_type", and "data"

    Returns:
      [type] -- [description]
    """
    dset_item = {}
    for feature in example.keys():
        dset_item[feature] = example[feature]["_type"](example[feature]["data"])
        example_proto = tf.train.Example(features=tf.train.Features(feature=dset_item))
    return example_proto.SerializeToString()

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def spec_to_record(spec, call_id, call_type):
    temp_spec = (norm(spec)*255).astype('uint8')
    example = serialize_example(
                                {
                                    "spectrogram": {
                                        "data": temp_spec.flatten().tobytes(),
                                        "_type": _bytes_feature,
                                    },
                                    "labels": {
                                        "data": np.bytes_(call_id),
                                        "_type": _bytes_feature,
                                    },
                                    "call_type": {
                                        "data": np.bytes_(call_type),
                                        "_type": _bytes_feature,
                                    },
                                    
                                }
                            )
    return example


class MDSAE(tf.keras.Model):
    """a basic autoencoder class for tensorflow
    Extends:
        tf.keras.Model
    """

    def __init__(self, **kwargs):
        super(MDSAE, self).__init__()
        self.__dict__.update(kwargs)

        self.enc = tf.keras.Sequential(self.enc)
        self.dec = tf.keras.Sequential(self.dec)

    @tf.function
    def encode(self, x):
        return self.enc(x)

    @tf.function
    def decode(self, z):
        return self.dec(z)

    @tf.function
    def compute_loss(self, x):
        z = self.encode(x)
        _x = self.decode(z)
        ae_loss = tf.reduce_mean(tf.square(x - _x))
        dist_loss = distance_loss(x, z)
        loss = ae_loss + dist_loss
        return loss, ae_loss, dist_loss

    @tf.function
    def compute_gradients(self, x):
        with tf.GradientTape() as tape:
            loss, _, _ = self.compute_loss(x)
        return tape.gradient(loss, self.trainable_variables)

    def train_net(self, train_x):
        gradients = self.compute_gradients(train_x)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


def plot_reconstruction(model, example_data, nex=5, zm=3):

    example_data_reconstructed = model.decode(model.encode(example_data))
    fig, axs = plt.subplots(ncols=nex, nrows=2, figsize=(zm * nex, zm * 2))
    for exi in range(nex):
        axs[0, exi].matshow(
            example_data.numpy()[exi].squeeze(), cmap=plt.cm.Greys  # , vmin=0, vmax=1
        )
        axs[1, exi].matshow(
            example_data_reconstructed.numpy()[exi].squeeze(),
            cmap=plt.cm.Greys,
            # vmin=0,
            # vmax=1,
        )
    for ax in axs.flatten():
        ax.axis("off")
    plt.show()


def distance_loss(x1, x2):
    """ Loss based on the distance between elements in a batch
    """
    sdx = squared_dist(tf.reshape(x1, [len(x1), tf.math.reduce_prod(tf.shape(x1)[1:])]))
    sdx = sdx / tf.reduce_mean(sdx)
    sdz = squared_dist(tf.reshape(x2, [len(x2), tf.math.reduce_prod(tf.shape(x2)[1:])]))
    sdz = sdz / tf.reduce_mean(sdz)
    return tf.reduce_mean(
        tf.square(
            tf.math.log(tf.constant(1.0) + sdx) - (tf.math.log(tf.constant(1.0) + sdz))
        )
    )


def squared_dist(A):
    """
    Computes the pairwise distance between points
    #http://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
    """
    expanded_a = tf.expand_dims(A, 1)
    expanded_b = tf.expand_dims(A, 0)
    distances = tf.reduce_mean(tf.math.squared_difference(expanded_a, expanded_b), 2)
    return distances

def _dtype_to_tf_feattype(dtype):
    """ convert tf dtype to correct tffeature format
    """
    if dtype in [tf.float32, tf.int64]:
        return dtype
    else:
        return tf.string

def _parse_function(example_proto, data_types):
    """ parse dataset from tfrecord, and convert to correct format
    """
    # list features
    features = {
        lab: tf.io.FixedLenFeature([], _dtype_to_tf_feattype(dtype))
        for lab, dtype in data_types.items()
    }
    # parse features
    parsed_features = tf.io.parse_single_example(example_proto, features)
    feat_dtypes = [tf.float32, tf.string, tf.int64]

    # convert the features if they are in the wrong format
    parse_list = [
        parsed_features[lab]
        if dtype in feat_dtypes
        else tf.io.decode_raw(parsed_features[lab], dtype)
        for lab, dtype in data_types.items()
    ]
    return parse_list
