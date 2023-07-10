# import dependencies
import numpy as np
import math
import re
import time
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds

# if __name__ == '__main__':

# Load dataset
print("1.1 Start loading files...")
with open('./europarl-v7.fr-en.en',
          mode='r',
          encoding='utf-8') as f:
    europar_en = f.read()

with open('./europarl-v7.fr-en.fr',
          mode='r',
          encoding='utf-8') as f:
    europar_de = f.read()

print("-------Completed loading files!")

print('1.2 cleaning data')
corpus_en = europar_en
# corpus_en = 'You  a  beautilful .A \n 131312312'
# print(corpus_en[:50])
corpus_en = re.sub('\.(?= [a-z]|[0-9]|[A-Z])', '.###', corpus_en)
corpus_en = re.sub(r'\.###', '', corpus_en)
corpus_en = re.sub(r'  +', ' ', corpus_en)
corpus_en = corpus_en.split('\n')
# print(corpus_en)
corpus_de = europar_de
# corpus_en = 'You  a  beautilful .A \n 131312312'
# print(corpus_en[:50])
corpus_de = re.sub('\.(?= [a-z]|[0-9]|[A-Z])', '.###', corpus_de)
corpus_de = re.sub(r'\.###', '', corpus_de)
corpus_de = re.sub(r'  +', ' ', corpus_de)
corpus_de = corpus_de.split('\n')
# print(europar_de)
print("-------Completed cleaning files!")

print('1.3 tokenizing text')

tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    corpus_en, target_vocab_size=2**13)
tokenizer_de = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    corpus_de, target_vocab_size=2**13)
tokenizer_en.save_to_file('./results/tokenizer_en')
tokenizer_de.save_to_file('./results/tokenizer_de')

VOCAB_SIZE_EN = tokenizer_en.vocab_size + 2  # 8189
VOCAB_SIZE_DE = tokenizer_de.vocab_size + 2  # 8170

print(f'VOCAB_SIZE_EN: {VOCAB_SIZE_EN}')
print(f'VOCAB_SIZE_DE: {VOCAB_SIZE_DE}')
inputs = [[VOCAB_SIZE_EN-2] + tokenizer_en.encode(sentence) + [VOCAB_SIZE_EN-1]
          for sentence in corpus_en]

outputs = [[VOCAB_SIZE_DE-2] + tokenizer_de.encode(sentence) + [VOCAB_SIZE_DE-1]
           for sentence in corpus_de]

print('completing tokenizing text')

print('1.4  Remove too long sentences')
MAX_LENGTH = 20

# remove inputs for length > 20
idx_to_remove = [count for count, sent in enumerate(inputs)
                 if len(sent) > MAX_LENGTH]

# for idx in reversed(idx_to_remove):
#     del corpus_en[idx]

for idx in reversed(idx_to_remove):
    del inputs[idx]
    del outputs[idx]

# remove outputs for length > 20
idx_to_remove = [count for count, sent in enumerate(outputs)
                 if len(sent) > MAX_LENGTH]
for idx in reversed(idx_to_remove):
    del inputs[idx]
    del outputs[idx]


print('Completing removing the long sentences')

# 2 - inputs/outputs creation
print('2 create inputs/outputs')

# make sure the inputs have the same length
inputs = tf.keras.utils.pad_sequences(inputs,
                                      value=0,
                                      padding='post',
                                      maxlen=MAX_LENGTH)

outputs = tf.keras.utils.pad_sequences(outputs,
                                       value=0,
                                       padding='post',
                                       maxlen=MAX_LENGTH)


print("start saving the inputs and outputs")
np.save('./inputs.npy', inputs)
np.save('./outputs.npy', outputs)


# load inputs and outputs

# inputs = np.load('./inputs.npy')
# outputs = np.load('./outputs.npy')

# print("Completing saving the inputs and outputs")


# print('2.1...completing transforming the inputs/outputs')
# BATCH_SIZE = 64
# BUFFER_SIZE = 20000
# dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))
# print('2.2...completing reading the inputs/outputs into dataset')
# dataset = dataset.cache()
# dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
# dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

"""

Testing for slices 
for index, sent in enumerate(corpus_de):
    if index == 10:
        break
    print(sent)
for index, sent in enumerate(outputs):
    if index == 10:
        break
    print(sent)
for index, sent in enumerate(outputs_sq):
    if index == 10:
        break
    print(sent)
"""