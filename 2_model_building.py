# import dependencies
import numpy as np
import math
import re
import time
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds


def scaled_fot_product_attention(queries, keys, values, mask):

    product = tf.matmul(queries, keys, transpose_b=True)
    key_dim = tf.cast(tf.shape(keys)[-1], tf.float32)
    scaled_product = product / tf.math.sqrt(key_dim)

    # explain why
    if mask is not None:
        scaled_product += (mask * -1e9)

    attention = tf.matmul(tf.nn.softmax(scaled_product, axis=-1), values)
    return attention


class MultiHeadAttention(layers.Layer):

    def __init__(self, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads

    def build(self, input_shape):
        self.d_model = input_shape[-1]
        assert self.d_model % self.n_heads == 0
        self.d_head = self.d_model // self.n_heads

        self.query_lin = layers.Dense(units=self.d_model)
        self.key_lin = layers.Dense(units=self.d_model)
        self.value_lin = layers.Dense(units=self.d_model)

        self.final_lin = layers.Dense(units=self.d_model)

    """
    在Transformer模型中，维度（dimension）和头（head）之间存在一种关系。

    维度（dimension）通常指的是输入向量的维度或隐藏层的维度，通常表示为d_model。在Transformer中，d_model代表了输入向量的维度，也是每个位置编码的维度。它确定了模型中神经网络层的大小。

    头（head）是指Transformer模型中的多头注意力（multi-head attention）机制。注意力机制的目的是允许模型在不同的表示空间中同时关注不同的信息。多头注意力机制通过将输入进行多次线性变换，并将这些变换结果分配给不同的头来实现。

    在Transformer中，输入向量首先通过线性变换映射到多个头（head）的维度上。然后，每个头独立进行注意力计算，生成每个头的输出结果。最后，这些头的输出结果通过线性变换和拼接（concatenate）操作，组合为最终的模型输出。

    维度（dimension）和头（head）之间的关系在于，维度（d_model）会被平均分配到每个头（head）的维度上。假设输入向量的维度（d_model）为512，如果有8个注意力头（heads），则每个注意力头的维度为512/8 = 64。这意味着每个注意力头只关注输入向量的子空间，其中维度为64。

    通过多头注意力机制，Transformer模型能够在不同维度的表示空间中并行处理输入，并且每个头能够专注于不同的特征和关系。这有助于提高模型的表示能力和捕捉输入之间的长程依赖关系。
    
    向量的维度（dimension）与嵌入（embedding）密切相关。

    在机器学习和自然语言处理任务中，嵌入（embedding）是将高维度的离散数据（如单词、字符、类别等）映射到低维度的连续向量空间中的过程。嵌入表示可以捕捉到数据之间的语义和关系，从而更好地在机器学习模型中进行处理和学习。

    嵌入的维度通常是由问题和数据的特性来确定的。在自然语言处理任务中，常见的词嵌入维度为100、200、300等。较低的维度可以降低计算负担，但可能会损失一些细节和语义信息；较高的维度可以提供更丰富的信息，但也会增加计算开销。

    与嵌入密切相关的是模型的输入和输出维度。在Transformer模型中，输入和输出的维度通常与嵌入的维度相匹配。例如，如果使用300维的词嵌入作为输入，那么Transformer模型的输入维度通常也是300。类似地，模型的输出维度也会匹配任务的要求。

    通过使用适当维度的嵌入和与之匹配的输入维度，可以更好地在模型中保留和学习数据的语义信息。维度的选择对于模型的性能和表现起着重要的作用，因此在设计模型和进行实验时需要考虑嵌入和输入维度的选择。    
    """

    def split_proj(self, inputs, batch_size):  # inputs: (batch_size, seq_length, d_model)
        shape = (batch_size,
                 -1,
                 self.n_heads,
                 self.d_head)

        # outputs: (batch_size, seq_length, nb_proj, d_proj)
        splited_inputs = tf.reshape(inputs, shape=shape)
        # outputs: (batch_size, nb_proj, seq_length,  d_proj)
        return tf.transpose(splited_inputs, perm=[0, 2, 1, 3])

    def call(self, queries, keys, values, mask):

        batch_size = tf.shape(queries)[0]

        queries = self.query_lin(queries)
        keys = self.key_lin(keys)
        values = self.value_lin(values)

        queries = self.split_proj(queries, batch_size)
        keys = self.split_proj(keys, batch_size)
        values = self.split_proj(values, batch_size)

        attention = scaled_fot_product_attention(queries, keys, values, mask)

        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention,
                                      shape=(batch_size, -1, self.d_model))
        outputs = self.final_lin(concat_attention)
        return outputs


class PositionalEncoding(layers.Layer):

    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def get_angles(self, pos, i, d_model):  # pos: (seq_length, 1) i: (1, d_model)
        # 2*(i//2) => if i = 5 -> ans = 4
        angles = 1 / np.power(10000., (2*(i//2)) / np.float32(d_model))
        return pos * angles  # (seq_length, d_model)

    def call(self, inputs):
        # input shape batch_size, seq_length, d_model
        seq_length = inputs.shape.as_list()[-2]
        d_model = inputs.shape.as_list()[-1]
        # Calculate the angles given the input
        angles = self.get_angles(np.arange(seq_length)[:, np.newaxis],
                                 np.arange(d_model)[np.newaxis, :],
                                 d_model)
        # Calculate the positional encodings
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        # Expand the encodings with a new dimension
        pos_encoding = angles[np.newaxis, ...]

        return inputs + tf.cast(pos_encoding, tf.float32)


class EncoderLayer(layers.Layer):

    def __init__(self, FFN_units, n_heads, dropout_rate):
        super(EncoderLayer, self).__init__()

        self.FFN_units = FFN_units
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.d_model = input_shape[-1]

        self.multi_head_attention = MultiHeadAttention(self.n_heads)
        self.dropout_1 = layers.Dropout(rate=self.dropout_rate)
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)

        self.ffn1_relu = layers.Dense(units=self.FFN_units, activation="relu")
        self.ffn2 = layers.Dense(units=self.d_model)
        self.dropout_2 = layers.Dropout(rate=self.dropout_rate)
        self.norm_2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, mask, training):
        attention = self.multi_head_attention(inputs, inputs, inputs, mask)
        attention = self.dropout_1(attention, training=training)
        attention = self.norm_1(attention+inputs)

        outputs = self.ffn1_relu(attention)
        outputs = self.ffn2(outputs)
        outputs = self.dropout_2(outputs)

        outputs = self.norm_2(outputs + attention)
        return outputs


class Encoder(layers.Layer):

    def __init__(self,
                 n_layers,
                 FFN_units,
                 n_heads,
                 dropout_rate,
                 vocab_size,
                 d_model,
                 name="encoder"):
        super(Encoder, self).__init__(name=name)
        self.n_layers = n_layers
        self.d_model = d_model
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding()
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.enc_layers = [EncoderLayer(
            FFN_units, n_heads, dropout_rate) for _ in range(n_layers)]

    def call(self, inputs, mask, training):
        outputs = self.embedding(inputs)
        outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        outputs = self.pos_encoding(outputs)
        outputs = self.dropout(outputs, training)

        for i in range(self.n_layers):
            outputs = self.enc_layers[i](outputs, mask, training)
        return outputs


class DecoderLayer(layers.Layer):

    def __init__(self, FFN_units, n_heads, dropout_rate):
        super(DecoderLayer, self).__init__()
        self.FFN_units = FFN_units
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.d_model = input_shape[-1]
        self.multi_head_causal_attention_1 = MultiHeadAttention(self.n_heads)
        self.dropout_1 = layers.Dropout(rate=self.dropout_rate)
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)

        self.multi_head_enc_dec_attention_1 = MultiHeadAttention(self.n_heads)
        self.dropout_2 = layers.Dropout(rate=self.dropout_rate)
        self.norm_2 = layers.LayerNormalization(epsilon=1e-6)

        # Feed foward
        self.ffn1_relu = layers.Dense(units=self.FFN_units, activation="relu")
        self.ffn2 = layers.Dense(units=self.d_model)
        self.dropout_3 = layers.Dropout(rate=self.dropout_rate)
        self.norm_3 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, enc_outputs, mask_1, mask_2, training):
        attention = self.multi_head_causal_attention_1(
            inputs, inputs, inputs, mask_1)
        attention = self.dropout_1(attention, training)

        attention = self.norm_1(attention + inputs)
        attention_2 = self.multi_head_enc_dec_attention_1(queries=attention,
                                                          keys=enc_outputs,
                                                          values=enc_outputs,
                                                          mask=mask_2)
        attention_2 = self.dropout_2(attention_2, training)
        attention_2 = self.norm_2(attention_2 + attention)

        outputs = self.ffn1_relu(attention_2)
        outputs = self.ffn2(outputs)
        outputs = self.dropout_3(outputs, training)
        outputs = self.norm_3(outputs + attention_2)
        return outputs


class Decoder(layers.Layer):

    def __init__(self, n_layers, FFN_units, n_heads, dropout_rate, vocab_size, d_model, name="decoder"):
        super(Decoder, self).__init__(name=name)
        self.d_model = d_model
        self.n_layers = n_layers
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding()
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.dec_layers = [DecoderLayer(
            FFN_units, n_heads, dropout_rate) for _ in range(n_layers)]

    def call(self, inputs, enc_outputs, mask_1, mask_2, training):
        outputs = self.embedding(inputs)
        outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        outputs = self.pos_encoding(outputs)
        outputs = self.dropout(outputs, training)
        for i in range(self.n_layers):
            outputs = self.dec_layers[i](outputs,
                                         enc_outputs,
                                         mask_1,
                                         mask_2, training)

        return outputs


class Transformer(tf.keras.Model):
    def __init__(self,
                 vocab_size_enc,
                 vocab_size_dec,
                 d_model,
                 n_layer,
                 FFN_units,
                 n_heads,
                 dropout_rate,
                 name='transformer'):
        super(Transformer, self).__init__(name=name)
        self.encoder = Encoder(n_layers=n_layer,
                               FFN_units=FFN_units,
                               n_heads=n_heads,
                               dropout_rate=dropout_rate,
                               vocab_size=vocab_size_enc,
                               d_model=d_model)
        self.decoder = Decoder(n_layers=n_layer,
                               FFN_units=FFN_units,
                               n_heads=n_heads,
                               dropout_rate=dropout_rate,
                               vocab_size=vocab_size_dec,
                               d_model=d_model)
        self.last_linear = layers.Dense(units=vocab_size_dec)

    def create_padding_mask(self, seq):
        mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, seq):
        seq_len = tf.shape(seq)[1]
        look_ahead_mask = 1 - \
            tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return look_ahead_mask

    def call(self, enc_inputs, dec_inputs, training):
        enc_mask = self.create_padding_mask(enc_inputs)
        dec_mask_1 = tf.maximum(
            self.create_look_ahead_mask(dec_inputs),
            self.create_padding_mask(dec_inputs)
        )
        dec_mask_2 = self.create_padding_mask(enc_inputs)

        enc_outputs = self.encoder(enc_inputs, enc_mask, training)

        dec_outputs = self.decoder(dec_inputs,
                                   enc_outputs,
                                   dec_mask_1,
                                   dec_mask_2,
                                   training)
        outputs = self.last_linear(dec_outputs)
        return outputs
