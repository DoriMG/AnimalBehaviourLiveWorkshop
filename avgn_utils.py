import tensorflow as tf
import numpy as np

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
    example = serialize_example(
                                {
                                    "spectrogram": {
                                        "data": spec.flatten().tobytes(),
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
