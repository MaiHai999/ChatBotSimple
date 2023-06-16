

from tensorflow.keras.layers import Attention
import re
from underthesea import word_tokenize
import os, sys
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import array
from numpy import asarray
from numpy import zeros
from gensim.models import Word2Vec
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM,Input,Embedding,Bidirectional,Concatenate
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from tensorflow.keras.layers import AdditiveAttention
from tensorflow.keras.models import Model



##tiền xử lý dữ liệu
# Lấy đường dẫn thư mục hiện tại
current_directory = os.getcwd()

# Tạo đường dẫn đến tệp small_chat_box1.txt
data_path = os.path.join(current_directory, "small_chat_box1.txt")

with open(data_path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")


BATCH_SIZE = 64
EPOCHS = 50
LSTM_NODES =256
NUM_SENTENCES = 20000
MAX_SENTENCE_LENGTH = 25
MAX_NUM_WORDS = 20000
EMBEDDING_SIZE = 100

# Hàm chuẩn hoá câu
def standardize_data(row):
    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    row = re.sub(r"[\.,\?]+$-", "", row)
    # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
    row = row.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ") \
        .replace("-", " ").replace("?", " ")
    row = row.strip().lower()
    # word_tokenize
    #row = word_tokenize(row , format="text")
    return row

#xử lý dữ liệu và đưa vào mảng
encoder_input_texts = []
decoder_input_texts = []
decoder_output_texts = []

for line in lines:
    input_text ,output_text ,_  = line.split("__eou__")
    encoder_input_texts.append(standardize_data(input_text))
    decoder_input_texts.append("<sos> " + standardize_data(output_text) )
    decoder_output_texts.append(standardize_data(output_text) + " <eos>")


#tạo từ điển cho encoder
input_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS,filters='')
input_tokenizer.fit_on_texts(encoder_input_texts)
input_integer_seq = input_tokenizer.texts_to_sequences(encoder_input_texts)
word2idx_inputs = input_tokenizer.word_index
max_input_len = max(len(sen) for sen in input_integer_seq)

#tạo từ điển cho decoder
output_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
output_tokenizer.fit_on_texts(decoder_input_texts + decoder_output_texts)
output_integer_seq = output_tokenizer.texts_to_sequences(decoder_output_texts)
output_input_integer_seq = output_tokenizer.texts_to_sequences(decoder_input_texts)
word2idx_outputs = output_tokenizer.word_index
num_words_output = len(word2idx_outputs) + 1
max_out_len = max(len(sen) for sen in output_integer_seq)

#pading encoder
encoder_input_sequences = pad_sequences(input_integer_seq, maxlen=max_input_len)

#pading decoder
decoder_input_sequences = pad_sequences(output_input_integer_seq, maxlen=max_out_len, padding='post')
decoder_output_sequences = pad_sequences(output_integer_seq, maxlen=max_out_len, padding='post')

#khởi tạo ma trận embeding cho encoder
encoder_num_words = min(MAX_NUM_WORDS, len(word2idx_inputs) + 1)
decoder_num_words = min(MAX_NUM_WORDS, len(word2idx_outputs) + 1)



encoder_inputs = Input(shape=(max_input_len,))
enc_emb = Embedding(encoder_num_words, 1024,trainable=True)(encoder_inputs)

# Bidirectional lstm layer
enc_lstm1 = Bidirectional(LSTM(256,return_sequences=True,return_state=True))
encoder_outputs1, forw_state_h, forw_state_c, back_state_h, back_state_c = enc_lstm1(enc_emb)

final_enc_h = Concatenate()([forw_state_h,back_state_h])
final_enc_c = Concatenate()([forw_state_c,back_state_c])

encoder_states =[final_enc_h, final_enc_c]

encoder_model = Model(encoder_inputs, encoder_states)

# Set up the decoder.
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(decoder_num_words, 1024,trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)

#LSTM using encoder_states as initial state
decoder_lstm = LSTM(512, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)


#Attention Layer
#attention_result = AdditiveAttention(use_scale=True)([decoder_outputs, encoder_outputs1])



# Concat attention output and decoder LSTM output
#decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attention_result])

#Dense layer
decoder_dense = Dense(decoder_num_words, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)


# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)



model.summary()

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='../model_plot4a.png', show_shapes=True, show_layer_names=True)

print(encoder_num_words)
print(decoder_num_words)

#khởi tạo ma trận one hot
decoder_targets_one_hot = np.zeros((
        len(encoder_input_texts),
        max_out_len,
        num_words_output
    ),
    dtype='float32'
)

for i, d in enumerate(decoder_output_sequences):
    for t, word in enumerate(d):
        decoder_targets_one_hot[i, t, word] = 1




from keras.utils.vis_utils import plot_model
plot_model(model, to_file='../model_plot4a.png', show_shapes=True, show_layer_names=True)


#chạy model
r = model.fit(
    [encoder_input_sequences, decoder_input_sequences],
    decoder_targets_one_hot,
    batch_size=BATCH_SIZE,
    epochs= 25,
    validation_split=0.1,
)

#luu model

# Lấy đường dẫn thư mục hiện tại
current_directory = os.getcwd()

# Tên thư mục cần truy cập
folder_name = "seq2seq_auto_chax_box_bilstm_industry1"

# Tạo đường dẫn đến thư mục
folder_path = os.path.join(current_directory, folder_name)

model.save(folder_path)



#vẽ sơ đồ
pd.DataFrame(r.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 3) # set the vertical range to [0-1]
plt.show()


#chỉnh model để có thể predict
decoder_state_input_h = Input(shape=(None,))
decoder_state_input_c = Input(shape=(None,))

decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_inputs_single = Input(shape=(1,))

decoder_inputs_single_x = dec_emb_layer(decoder_inputs_single)

decoder_outputs1, h, c = decoder_lstm(decoder_inputs_single_x, initial_state=decoder_states_inputs)

decoder_states = [h, c]
decoder_outputs = decoder_dense(decoder_outputs1)
decoder_model = Model(
    [decoder_inputs_single] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

plot_model(decoder_model, to_file='model_plot4a.png', show_shapes=True, show_layer_names=True)

idx2word_input = {v:k for k, v in word2idx_inputs.items()}
idx2word_target = {v:k for k, v in word2idx_outputs.items()}



def translate_sentence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2idx_outputs['<sos>']
    eos = word2idx_outputs['<eos>']
    output_sentence = []
    for _ in range(max_out_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        idx = np.argmax(output_tokens[0, 0, :])

        if eos == idx:
            break

        word = ''

        if idx > 0:
            word = idx2word_target[idx]
            output_sentence.append(word)

        target_seq[0, 0] = idx
        states_value = [h, c]

    return ' '.join(output_sentence)

for i in range(20):
  i = np.random.choice(len(encoder_input_texts))
  input_seq = encoder_input_sequences[i:i+1]
  translation = translate_sentence(input_seq)
  print('-')
  print('Input:', encoder_input_texts[i])
  print('Response:', translation)



# i = np.random.choice(len(encoder_input_texts))
# i = 31
# input_seq = encoder_input_sequences[i:i+1]
# translation = translate_sentence(input_seq)
# print('-')
# print('Input:', encoder_input_texts[i])
# print('Response:', translation)




