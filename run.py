import tensorflow as tf
import pandas as pd
from pathlib import Path
import re
import numpy as np
import unicodedata
import io
import keras
from nltk.corpus import stopwords
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from transformers import *
from transformers import BertTokenizer, TFBertModel, BertConfig

TRAIN_FILE = Path(r'D:\nlp_dataset\kaggle_natural_disastsers\train.csv')
TEST_FILE = Path(r'D:\nlp_dataset\kaggle_natural_disastsers\test.csv')
EVAL_OUT = Path(r'D:\nlp_dataset\kaggle_natural_disastsers\evaluation.csv')


class CustomBertModel(tf.keras.Model):
    def __init__(self):
        super(CustomBertModel, self).__init__()
        self.bert = TFBertModel.from_pretrained('bert-base-uncased')
        self.dense_1 = keras.layers.Dense(32, activation='relu', name='dense1')
        self.dropout_1 = keras.layers.Dropout(0.2, name='dropout1')
        self.dense_2 = keras.layers.Dense(2, activation='softmax', name='dense2')

    def call(self, inputs):
        '''
        BERT output followed by a dense layer, a dropout layer and the final classification layer
        :param inputs:
        :return:
        '''
        output_pooling = self.bert(inputs)
        dense1_output = self.dense_1(output_pooling.pooler_output)
        dropout1_output = self.dropout_1(dense1_output)
        dense2_output = self.dense_2(dropout1_output)
        return dense2_output


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def clean_stopwords_shortwords(w):
    stopwords_list = stopwords.words('english')
    words = w.split()
    clean_words = [w for w in words if (w not in stopwords_list) and len(w) > 2]
    return " ".join(clean_words)


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = clean_stopwords_shortwords(w)
    w = re.sub(r'@\w+', '', w)
    return w


def load_data(train_file: Path, train: bool):
    lines = list()
    with open(train_file, 'rt', encoding='utf-8') as fin:
        for idx, line in enumerate(fin):
            line = line.strip('\n')
            if not re.search(r'^[0-9]+,', line) and idx != 0:
                lines[-1] += ' ' + line
            else:
                lines.append(line)
    f = io.StringIO('\n'.join(lines))
    df = pd.read_csv(f)
    if train:
        df = df.rename(columns={'target': 'label'}, inplace=False)
    df['text'] = df['text'].map(preprocess_sentence)
    if train:
        df = shuffle(df)

    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    sentences = df['text']
    if train:
        labels = df['label']
        assert len(sentences) == len(labels)
    input_ids = list()
    attention_masks = list()
    for sent in sentences:
        # encode_plus() outputs the encoded sentence, with [CLS] and [SEP]
        bert_input = bert_tokenizer.encode_plus(sent, add_special_tokens=True, max_length=64, pad_to_max_length=True,
                                                return_attention_mask=True)
        input_ids.append(bert_input['input_ids'])
        attention_masks.append(bert_input['attention_mask'])

    input_ids = np.asarray(input_ids)
    attention_masks = np.array(attention_masks)
    if train:
        labels = np.array(labels)
        return (df, input_ids, attention_masks, labels)
    else:
        return (df, input_ids, attention_masks)


def main():
    train_df, input_ids, attention_masks, labels = load_data(TRAIN_FILE, True)
    train_inp, val_inp, train_label, val_label, train_mask, val_mask = train_test_split(input_ids, labels,
                                                                                        attention_masks, test_size=0.2)
    print(f"Training data: {train_label.shape}")
    print(f"Validation data: {val_label.shape}")
    model_save_path = Path(r'output\natural_disaster')
    model_save_path.mkdir(exist_ok=True)
    saved_model_path = model_save_path / "saved_model.pb"
    saved_weights_path = model_save_path / "saved_weights.weights"

    bert_model = CustomBertModel()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss")
    ]

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08)
    bert_model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

    history = bert_model.fit([train_inp, train_mask], train_label, batch_size=16, epochs=4,
                             validation_data=([val_inp, val_mask], val_label), callbacks=callbacks)
    bert_model.save_weights(saved_weights_path)

    trained_model = CustomBertModel()
    trained_model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
    trained_model.load_weights(saved_weights_path)

    # Test custom model
    test_df, test_input_ids, test_attention_masks = load_data(TEST_FILE, False)
    preds = trained_model.predict([test_input_ids, test_attention_masks], batch_size=32)
    pred_labels = np.argmax(preds, axis=1)
    test_df['target'] = pred_labels
    test_df[['id', 'target']].to_csv(EVAL_OUT, index=False)


if __name__ == '__main__':
    main()

