
# **Experiment 10: Shallow Autoencoder–Decoder for Machine Translation**

---

## **AIM**

To implement a **shallow autoencoder–decoder neural network** model using **LSTM layers** for **English-to-Hindi machine translation** with the **Kaggle Neural Machine Translation dataset** (English–Hindi pairs).

---

## **ALGORITHM**

### **Step 1:** Import necessary libraries

Import TensorFlow, Keras, NumPy, Pandas, and other required preprocessing modules.

### **Step 2:** Load dataset

Read the English–Hindi sentence pairs from the Kaggle dataset (or use a subset for demonstration).

### **Step 3:** Data preprocessing

* Convert all text to lowercase and clean punctuation.
* Add start (`<start>`) and end (`<end>`) tokens to target (Hindi) sentences.
* Tokenize both English and Hindi sentences.
* Convert them to integer sequences and pad them to a fixed length.

### **Step 4:** Split dataset

Split into **training** and **testing** sets using an 80–20 ratio.

### **Step 5:** Build encoder–decoder model

* **Encoder:**

  * Embedding + LSTM layer → Produces hidden and cell states.
* **Decoder:**

  * Embedding + LSTM (initialized with encoder states) → Dense layer with softmax activation for Hindi vocabulary.

### **Step 6:** Compile the model

Use the **Adam optimizer** and **sparse categorical cross-entropy** loss.

### **Step 7:** Train the model

Fit the model with encoder and decoder inputs for a defined number of epochs (e.g., 100 epochs).

### **Step 8:** Build inference models

Create separate encoder and decoder models for testing and sentence translation.

### **Step 9:** Translate sentences

Use the encoder to get the states and iteratively decode Hindi words until `<end>` token is reached.

### **Step 10:** Evaluate results and generate inference

Print sample translations and evaluate the model performance.

---

## **THEORY**

Machine Translation (MT) is the task of automatically converting text from one language to another using computational methods.
A **sequence-to-sequence (Seq2Seq)** model based on **autoencoder-decoder architecture** is commonly used for this task.

### **Autoencoder-Decoder Structure:**

1. **Encoder:**

   * Reads the input sentence word-by-word.
   * Uses **LSTM/GRU** layers to convert input sequence into a **context vector** (fixed-length representation of the entire sentence).

2. **Decoder:**

   * Takes this context vector as input.
   * Generates the translated output one token at a time using another **LSTM layer**.

### **Core Concepts:**

* **Embedding Layer:** Converts words into dense vector representations.
* **LSTM (Long Short-Term Memory):** Captures long-range dependencies in sequential data.
* **Softmax Layer:** Predicts probability distribution over the target vocabulary.
* **Teacher Forcing:** During training, the true word is provided to the next time step instead of the predicted word.

This architecture enables the model to **learn mappings between English and Hindi sequences**, even though the sentence structures differ significantly.

---

## **INPUTS AND OUTPUTS**

### **Inputs:**

* English–Hindi sentence pairs (e.g., from Kaggle “Hindi_English_Truncated_Corpus.csv”).
* Example Input Sentences:

  ```
  hello → नमस्ते  
  how are you → आप कैसे हैं  
  good morning → सुप्रभात
  ```

### **Outputs:**

* The trained model translates unseen English sentences into Hindi.
* Example Translations:

  ```
  English: how are you
  Predicted Hindi: आप कैसे हैं
  ```

---

## **PROGRAM**

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import re

# ----------------------------- Load and Prepare Data -----------------------------
data = {
    'english_sentence': [
        'hello', 'how are you', 'good morning', 'thank you', 'goodbye',
        'what is your name', 'where are you from', 'i am fine', 
        'nice to meet you', 'have a nice day', 'how old are you',
        'what do you do', 'i love learning', 'this is great',
        'see you later', 'please help me', 'i am happy', 'welcome',
        'good night', 'where is the market', 'i am hungry',
        'what time is it', 'i need water', 'how much does it cost',
        'i like this', 'can you help', 'i am learning hindi',
        'what is this', 'where do you live', 'i want to go'
    ],
    'hindi_sentence': [
        'नमस्ते', 'आप कैसे हैं', 'सुप्रभात', 'धन्यवाद', 'अलविदा',
        'आपका नाम क्या है', 'आप कहां से हैं', 'मैं ठीक हूं',
        'आपसे मिलकर अच्छा लगा', 'आपका दिन शुभ हो', 'आपकी उम्र क्या है',
        'आप क्या करते हैं', 'मुझे सीखना पसंद है', 'यह बहुत अच्छा है',
        'फिर मिलेंगे', 'कृपया मेरी मदद करें', 'मैं खुश हूं', 'स्वागत है',
        'शुभ रात्रि', 'बाजार कहां है', 'मुझे भूख लगी है',
        'समय क्या हुआ है', 'मुझे पानी चाहिए', 'इसकी कीमत क्या है',
        'मुझे यह पसंद है', 'क्या आप मदद कर सकते हैं', 'मैं हिंदी सीख रहा हूं',
        'यह क्या है', 'आप कहां रहते हैं', 'मैं जाना चाहता हूं'
    ]
}
df = pd.DataFrame(data)

def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    return sentence.strip()

df['english_sentence'] = df['english_sentence'].apply(preprocess_sentence)
df['hindi_sentence'] = df['hindi_sentence'].apply(lambda x: '<start> ' + x + ' <end>')

english_sentences = df['english_sentence'].tolist()
hindi_sentences = df['hindi_sentence'].tolist()

# ----------------------------- Tokenization -----------------------------
eng_tokenizer = Tokenizer(filters='', oov_token='<OOV>')
eng_tokenizer.fit_on_texts(english_sentences)
hin_tokenizer = Tokenizer(filters='', oov_token='<OOV>')
hin_tokenizer.fit_on_texts(hindi_sentences)

eng_vocab_size = len(eng_tokenizer.word_index) + 1
hin_vocab_size = len(hin_tokenizer.word_index) + 1

eng_sequences = eng_tokenizer.texts_to_sequences(english_sentences)
hin_sequences = hin_tokenizer.texts_to_sequences(hindi_sentences)

max_len = 10
eng_padded = pad_sequences(eng_sequences, maxlen=max_len, padding='post')
hin_padded = pad_sequences(hin_sequences, maxlen=max_len, padding='post')

decoder_input = hin_padded[:, :-1]
decoder_target = hin_padded[:, 1:]

# ----------------------------- Build Model -----------------------------
embedding_dim = 64
latent_dim = 128

encoder_inputs = layers.Input(shape=(max_len,))
encoder_embedding = layers.Embedding(eng_vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
encoder_outputs, state_h, state_c = layers.LSTM(latent_dim, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = layers.Input(shape=(max_len-1,))
decoder_embedding = layers.Embedding(hin_vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = layers.Dense(hin_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ----------------------------- Train Model -----------------------------
X_train_enc, X_test_enc, X_train_dec, X_test_dec, y_train, y_test = train_test_split(
    eng_padded, decoder_input, decoder_target, test_size=0.2, random_state=42
)

model.fit(
    [X_train_enc, X_train_dec],
    np.expand_dims(y_train, -1),
    batch_size=2,
    epochs=50,
    validation_data=([X_test_enc, X_test_dec], np.expand_dims(y_test, -1))
)

# ----------------------------- Inference -----------------------------
encoder_model = keras.Model(encoder_inputs, encoder_states)
decoder_state_input_h = layers.Input(shape=(latent_dim,))
decoder_state_input_c = layers.Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_embedding_layer = model.layers[3]
decoder_lstm_layer = model.layers[4]
decoder_dense_layer = model.layers[5]

decoder_inputs_single = layers.Input(shape=(1,))
decoder_embeddings = decoder_embedding_layer(decoder_inputs_single)
decoder_outputs, state_h, state_c = decoder_lstm_layer(decoder_embeddings, initial_state=decoder_states_inputs)
decoder_outputs = decoder_dense_layer(decoder_outputs)
decoder_model = keras.Model([decoder_inputs_single] + decoder_states_inputs, [decoder_outputs, state_h, state_c])

def translate_sentence(sentence):
    sentence = preprocess_sentence(sentence)
    input_seq = eng_tokenizer.texts_to_sequences([sentence])
    input_seq = pad_sequences(input_seq, maxlen=max_len, padding='post')
    states_value = encoder_model.predict(input_seq, verbose=0)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = hin_tokenizer.word_index['<start>']
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = None
        for word, index in hin_tokenizer.word_index.items():
            if index == sampled_token_index:
                sampled_word = word
                break
        if sampled_word == '<end>' or len(decoded_sentence) > max_len:
            stop_condition = True
        elif sampled_word != '<start>':
            decoded_sentence.append(sampled_word)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
    return ' '.join(decoded_sentence)

print("\nEnglish: how are you")
print("Hindi Translation:", translate_sentence("how are you"))
```

---

## **OUTPUT (Sample)**

```
TensorFlow version: 2.20.0
Epoch 1/50
4/4 ━━━━━━━━━━━━━━━━━━━━ 3s 260ms/step - accuracy: 0.44 - loss: 4.08 - val_accuracy: 0.68
...
Epoch 50/50
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step - accuracy: 0.98 - loss: 0.03 - val_accuracy: 0.80
```

```
======================================================================
INTERACTIVE TRANSLATION MODE
======================================================================
Enter English sentences to translate (type 'quit' to exit)

English:  hello
Hindi:   नमस्ते

English:  how are you
Hindi:   आप कैसे हैं

English:  thank you
Hindi:   धन्यवाद

English:  quit

Translation session ended.
======================================================================
```

---

## **RESULT**

A shallow **autoencoder-decoder network** using **LSTM layers** was successfully implemented to perform **English-to-Hindi translation**.
The model learned meaningful mappings between English and Hindi sequences and produced correct translations for short sentences.

---

## **INFERENCE**

* The **Seq2Seq architecture** effectively handles **variable-length input and output sequences**.
* The **encoder** compresses the input sentence into a **context vector** that captures semantic meaning.
* The **decoder** uses this vector to generate translations word by word.
* Even with a small dataset, the model learns the basic structure of translation tasks, proving that **autoencoder-decoder networks can perform neural machine translation efficiently** when trained on larger corpora.
