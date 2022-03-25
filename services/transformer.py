import os
import pickle

import numpy as np
import tensorflow as tf

from .architecture import Transformer

image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

num_layer = 4
d_model = 512
dff = 2048
num_heads = 8
row_size = 8
col_size = 8
top_k = 5000
target_vocab_size = top_k + 1
dropout_rate = 0.1

transformer = Transformer(
    num_layer,
    d_model,
    num_heads,
    dff,
    row_size,
    col_size,
    target_vocab_size,
    max_pos_encoding=target_vocab_size,
    rate=dropout_rate
)

transformer.load_weights('./model/transformer/checkpoints/my_checkpoint')

with open('./model/transformer/word_index.pickle', 'rb') as handle:
    word_index = pickle.load(handle)
with open('./model/transformer/index_word.pickle', 'rb') as handle:
    index_word = pickle.load(handle)
with open('./model/transformer/img_to_cap.pickle', 'rb') as handle:
    img_to_cap = pickle.load(handle)
with open('./model/transformer/img_name_val.pickle', 'rb') as handle:
    img_name_val = pickle.load(handle)
with open('./model/transformer/cap_val.pickle', 'rb') as handle:
    cap_val = pickle.load(handle)


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    # Delete temp file
    if '/temp/' in image_path:
        os.remove(image_path)
    return img, image_path


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_masks_decoder(tar):
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return combined_mask


def evaluate_transformer(image):

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    start_token = word_index['<start>']
    end_token = word_index['<end>']

    #decoder input is start token.
    decoder_input = [start_token]
    output = tf.expand_dims(decoder_input, 0) #tokens
    result = [] #word list

    for i in range(100):
        dec_mask = create_masks_decoder(output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(img_tensor_val,output,False,dec_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        # return the result if the predicted_id is equal to the end token
        if predicted_id == end_token:
          return result,tf.squeeze(output, axis=0), attention_weights
        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        result.append(index_word[int(predicted_id)])
        output = tf.concat([output, predicted_id], axis=-1)

    return (result, tf.squeeze(output, axis=0), attention_weights)