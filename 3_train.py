# import dependencies
import numpy as np
import math
import re
import time
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
from model_building import *


print("Hello world! you are starting to load dataset!")
inputs = np.load('./inputs.npy')
outputs = np.load('./outputs.npy')

print("Completing saving the inputs and outputs")


print('2.1...completing transforming the inputs/outputs')
# Hyper-parameters:
BATCH_SIZE = 64  # 64 # Training Batch
BUFFER_SIZE = 20000  # For shuffle size
D_MODEL = 128  # 512, 128. The dimension of embedding the word
N_LAYERS = 4  # The number of layer (N)
FFN_UNITS = 512  # 2048, 512. The unit of the Dense Layer in Multi-Head Attention
N_HEADS = 8  # The number of Multi-Head Attention
DROPOUT_RATE = 0.1
VOCAB_SIZE_EN = 8189  # The vocabulary size of English
VOCAB_SIZE_DE = 8170  # The vocabulary size of French
MAX_LENGTH = 20
EPOCHES = 10  # Training Epoches

# loading data
dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))
print('2.2...completing reading the inputs/outputs into dataset')
dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

# Build
transformer = Transformer(
    vocab_size_enc=VOCAB_SIZE_EN,
    vocab_size_dec=VOCAB_SIZE_DE,
    d_model=D_MODEL,
    n_layer=N_LAYERS,
    FFN_units=FFN_UNITS,
    n_heads=N_HEADS,
    dropout_rate=DROPOUT_RATE,
)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,
    reduction="none",
)


def loss_function(target, pred):
    mask = tf.math.logical_not(tf.math.equal(target, 0))
    loss_ = loss_object(target, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        # it is really import to cast into tf.float 32 to train the models
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(D_MODEL)
optimizer = tf.keras.optimizers.Adam(learning_rate,
                                     beta_1=0.9,
                                     beta_2=0.98,
                                     epsilon=1e-9)


checkpoint_path = './results'

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!')


for epoch in range(EPOCHES):
    print("Start of epoch {}".format(epoch+1))
    start = time.time()
    train_loss.reset_states()
    train_accuracy.reset_states()
    for (batch, (enc_inputs, targets)) in enumerate(dataset):
        dec_inputs = targets[:, :-1]
        dec_outputs_real = targets[:, 1:]
        with tf.GradientTape() as tape:
            prediction = transformer(enc_inputs, dec_inputs, True)
            loss = loss_function(dec_outputs_real, prediction)
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, transformer.trainable_variables))
        train_loss(loss)
        train_accuracy(dec_outputs_real, prediction)
        if batch % 50 == 0:
            print('Epoch {} Batch {} Loss: {:.4f} Accuracy: {:.4f}'.format(
                epoch+1, batch, train_loss.result(), train_accuracy.result()
            ))
    ckpt_save_path = ckpt_manager.save()
    print('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
    print('Time taken for 1 epoch: {} sec\n'.format(time.time() - start))


