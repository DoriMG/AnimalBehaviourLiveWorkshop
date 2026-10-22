# -*- coding: utf-8 -*-
"""
Script to train an AE
@author: Dori Grijseels, modified from avgn (https://github.com/timsainb/avgn_paper) by Tim Sainburg
"""


import pandas as pd
from avgn_utils import _parse_function, MDSAE

import tensorflow as tf
import numpy as np

from tqdm.autonotebook import tqdm
from IPython import display
from pathlib2 import Path




# Data and model locations
record_loc =  r"marmoset_calls.tfrecords"
save_loc = 'AE_model_marmoset.keras'

# Constants for model training
n_records = 2020 # Hardcoded here, but better to extract from your dataframe
BATCH_SIZE = 128 # Batch size for training
DIMS = (128, 128, 1) # Spectrogram size
TRAIN_SIZE=round(n_records*0.8)
TEST_SIZE=round(n_records*0.2)
TRAIN_BUF = 1010
TEST_BUF = 1010
n_epochs = 10

# Calculate batch numbers
N_TRAIN_BATCHES =int((TRAIN_SIZE-TEST_SIZE)/BATCH_SIZE)
N_TEST_BATCHES = int(TEST_SIZE/BATCH_SIZE)

# Model architecture
N_Z = 32 # Number of dimensions
encoder = [
    tf.keras.layers.InputLayer(input_shape=DIMS),
    tf.keras.layers.Conv2D(
        filters=32, kernel_size=3, strides=(2, 2), activation="relu"
    ),
    tf.keras.layers.Conv2D(
        filters=64, kernel_size=3, strides=(2, 2), activation="relu"
    ),
    tf.keras.layers.Conv2D(
        filters=128, kernel_size=3, strides=(2, 2), activation="relu"
    ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=N_Z),
]

decoder = [
    tf.keras.layers.Dense(units=16 * 16 * 256, activation="relu"),
    tf.keras.layers.Reshape(target_shape=(16, 16, 256)),
    tf.keras.layers.Conv2DTranspose(
        filters=128, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
    ),
    tf.keras.layers.Conv2DTranspose(
        filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
    ),
    tf.keras.layers.Conv2DTranspose(
        filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
    ),
    tf.keras.layers.Conv2DTranspose(
        filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation="tanh"
    ),
]

# Load data
raw_dataset = tf.data.TFRecordDataset([record_loc])

data_types = {
    "spectrogram": tf.uint8,
    "labels": tf.string,
    "call_type": tf.string
}

# parse each data type to the raw dataset
dataset = raw_dataset.map(lambda x: _parse_function(x, data_types=data_types))

# Divide into test and train
test_dataset = dataset.take(TEST_SIZE).shuffle(TEST_BUF).batch(BATCH_SIZE)
train_dataset = dataset.skip(TEST_SIZE).take(TRAIN_SIZE-TEST_SIZE).shuffle(TEST_BUF).batch(BATCH_SIZE)

 
        
# the optimizer for the model
optimizer = tf.keras.optimizers.Adam(1e-3)
# train the model
model = MDSAE(
    enc = encoder,
    dec = decoder,
    optimizer = optimizer,
)

model.build(DIMS)
    
# exampled data for plotting results
example_data = next(iter(test_dataset))
example_data = (
        tf.cast(tf.reshape(example_data[0], [BATCH_SIZE] + list(DIMS)), tf.float32)
        / 255
)
    
# a pandas dataframe to save the loss information to
losses = pd.DataFrame(columns = ['loss','recon_loss', 'distance_loss'])

for epoch in range(n_epochs):
    # train
    for batch, train_x in tqdm(
        zip(range(N_TRAIN_BATCHES), train_dataset), total=N_TRAIN_BATCHES
    ):
        #print(train_x[-1].numpy()[:4])
        x = tf.cast(tf.reshape(train_x[0], [BATCH_SIZE] + list(DIMS)), tf.float32) / 255
        model.train_net(x)
    # test on holdout
    loss = []
    for batch, test_x in tqdm(
        zip(range(N_TEST_BATCHES), test_dataset), total=N_TEST_BATCHES
    ):
        x = tf.cast(tf.reshape(test_x[0], [BATCH_SIZE] + list(DIMS)), tf.float32) / 255
        loss.append(model.compute_loss(x))
    losses.loc[len(losses)] = np.mean(loss, axis=0)
    # plot results
    display.clear_output()
    print(
        "Epoch: {} | recon_loss: {}".format(
            epoch, losses.recon_loss.values[-1]
        )
    )


# Save the entire model to a HDF5 file.
weight_save_file = str(epoch).zfill(4)+'.weights.h5'
model.save_weights(weight_save_file) 
model_save_file = save_loc
model.save(model_save_file)

# Extract the latent features by running only the encode part
latent_dataset = []
for i, record in enumerate(tqdm(dataset.as_numpy_iterator())):
    # Loop through dataset and find FP/calls
        x = tf.cast(np.reshape(record[0], [1]+ list(DIMS)), tf.float32) / 255
        latent = model.encode(x).numpy()
            
        entry = pd.DataFrame.from_dict({'call_id': [record[1].decode("utf-8")],     
                                         "call_type": [record[2].decode("utf-8")]})
        
        n_latent_means = latent.shape[1]
        for i in range(n_latent_means):
        	entry[f'latent_mean_{i}'] = latent[:, i]

        latent_dataset +=[entry]


latent_dataset = pd.concat(latent_dataset, ignore_index=True) 
latent_dataset.to_csv('MDSAE_latent_features.csv', index=False)
