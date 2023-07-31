import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformer_model import build_transformer_model  # Import build_transformer_model function

# Step 1: Load the Dataset
data_df = pd.read_csv('C:\\Users\\Manish Acharya\\OneDrive\\Desktop\\Major\\spoc-train.tsv', delimiter='\t')

# Step 2: Handle Missing Values
data_df = data_df.dropna(subset=['text', 'code'])  # Drop rows with missing values in 'text' or 'code'

# Step 3: Tokenization and Vocabulary Creation
tokenizer = Tokenizer(filters='', lower=False, oov_token='<UNK>')
combined_text = list(data_df['text']) + list(data_df['code'])
tokenizer.fit_on_texts(combined_text)

# Vocabulary mapping: token to index
vocab = tokenizer.word_index
# Add special tokens to the vocabulary
special_tokens = {'<PAD>': 0, '<SOS>': len(vocab) + 1, '<EOS>': len(vocab) + 2}
vocab.update(special_tokens)
# Reverse vocabulary mapping: index to token
reverse_vocab = {index: token for token, index in vocab.items()}

# Step 4: Numerical Representation and Padding
pseudocode_sequences = tokenizer.texts_to_sequences(data_df['text'])
cpp_code_sequences = tokenizer.texts_to_sequences(data_df['code'])

# Pad sequences to have the same length
max_sequence_length = max(len(seq) for seq in pseudocode_sequences + cpp_code_sequences)
padded_pseudocode = pad_sequences(pseudocode_sequences, maxlen=max_sequence_length, padding='post')
padded_cpp_code = pad_sequences(cpp_code_sequences, maxlen=max_sequence_length, padding='post')

# Step 5: Data Splitting
train_pseudocode, val_test_pseudocode, train_cpp_code, val_test_cpp_code = train_test_split(
    padded_pseudocode, padded_cpp_code, test_size=0.2, random_state=42
)
val_pseudocode, test_pseudocode, val_cpp_code, test_cpp_code = train_test_split(
    val_test_pseudocode, val_test_cpp_code, test_size=0.5, random_state=42
)

# Step 6: Save Preprocessed Data
with open('vocab.txt', 'w') as file:
    for token, index in vocab.items():
        file.write(f"{token}\t{index}\n")

np.save('train_pseudocode.npy', train_pseudocode)
np.save('val_pseudocode.npy', val_pseudocode)
np.save('test_pseudocode.npy', test_pseudocode)
np.save('train_cpp_code.npy', train_cpp_code)
np.save('val_cpp_code.npy', val_cpp_code)
np.save('test_cpp_code.npy', test_cpp_code)
