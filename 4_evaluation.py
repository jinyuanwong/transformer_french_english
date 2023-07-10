# import dependencies
import os
import numpy as np
import math
import re
import time
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
from model_building import *

# Hyper-parameters:
D_MODEL = 128  # 512, 128
N_LAYERS = 4  # 6
FFN_UNITS = 512  # 2048, 512
N_HEADS = 8  # 8
DROPOUT_RATE = 0.1

VOCAB_SIZE_EN = 8189
VOCAB_SIZE_DE = 8170
MAX_LENGTH = 20
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
    status = ckpt.restore(ckpt_manager.latest_checkpoint)
    status.expect_partial()
    print('Latest checkpoint restored!')


def evaluate(input_sentence):
    input_sentence = \
        [VOCAB_SIZE_EN - 2] + \
        tokenizer_en.encode(input_sentence) + [VOCAB_SIZE_EN-1]

    enc_input = tf.expand_dims(input_sentence, axis=0)

    output = tf.expand_dims([VOCAB_SIZE_DE-2], axis=0)
    for _ in range(MAX_LENGTH):
        # (1, seq_length, vocab_de)
        predictions = transformer(enc_input, output, False)

        prediction = predictions[:, -1:, :]
        prediction_id = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)

        if prediction_id == VOCAB_SIZE_DE-1:
            return tf.squeeze(output, axis=0)

        output = tf.concat([output, prediction_id], axis=-1)
    return (tf.squeeze(output, axis=0))


def translate(sentence):
    output = evaluate(sentence).numpy()
    predicted_sentence = tokenizer_de.decode(
        [i for i in output if i < VOCAB_SIZE_DE-2])

    print(f'Input: {sentence}')
    print(f'Output: {predicted_sentence}')


tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
    './dataset/tokenizer_en')
tokenizer_de = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
    './dataset/tokenizer_de')

if __name__ == '__main__':
    # arg = os.argv
    # if len(arg) > 1:
    #     translate(arg[1:])
    # else:
    translate("I want to travel!")
    translate("I need to practice so I can master.")
    translate("Future is what you can dream for.")
    translate("Exercising makes you healhty.")
    translate("This is a problem that I must solve.")
    translate("I hope you have a good day.")
