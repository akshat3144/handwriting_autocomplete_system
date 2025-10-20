"""
Handwritten Text Recognition Using Deep Learning: A CNN-LSTM Approach
Based on the research paper implementation for IAM Dataset
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

# ============================================================================
# 1. PREPROCESSING FUNCTIONS
# ============================================================================

def grayscale_conversion(image):
    """Convert RGB image to grayscale using weighted sum"""
    if len(image.shape) == 3:
        # Formula: I_gray = 0.2989*R + 0.5870*G + 0.1140*B
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return gray


def binarization(gray_image, method='otsu'):
    """Apply thresholding to convert grayscale to binary"""
    if method == 'otsu':
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'huang':
        threshold = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)[0]
        _, binary = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    return binary


def resize_and_pad(image, target_height=32, target_width=128):
    """Resize image to fixed dimensions with padding"""
    h, w = image.shape[:2]
    
    # Calculate scaling factor to maintain aspect ratio
    scale = min(target_height / h, target_width / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create padded image with white background
    padded = np.ones((target_height, target_width), dtype=np.uint8) * 255
    
    # Calculate padding offsets
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    
    # Place resized image in center
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded


def image_inversion(binary_image):
    """Invert pixel intensities (black text on white becomes white text on black)"""
    inverted = 255 - binary_image
    return inverted


def normalize_image(image):
    """Normalize pixel values to range [0, 1]"""
    normalized = image.astype(np.float32) / 255.0
    return normalized


def preprocess_image(image_path, target_height=32, target_width=128):
    """Complete preprocessing pipeline"""
    # Read image
    image = cv2.imread(str(image_path))  # Convert Path to string
    
    # Check if image was loaded successfully
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    # Step 1: Grayscale conversion
    gray = grayscale_conversion(image)
    
    # Step 2: Binarization
    binary = binarization(gray, method='otsu')
    
    # Step 3: Resize and pad
    resized = resize_and_pad(binary, target_height, target_width)
    
    # Step 4: Image inversion (optional, depending on dataset)
    # inverted = image_inversion(resized)
    
    # Step 5: Normalization
    normalized = normalize_image(resized)
    
    # Add channel dimension
    preprocessed = np.expand_dims(normalized, axis=-1)
    
    return preprocessed


# ============================================================================
# 2. CHARACTER ENCODING
# ============================================================================

class CharacterEncoder:
    """Encode and decode characters for model training"""
    
    def __init__(self, characters=None):
        if characters is None:
            # Default character set (lowercase + uppercase + digits + space)
            self.characters = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'-"
        else:
            self.characters = characters
        
        # Create character to index mapping (start from 0)
        # CTC blank token will be at index len(characters) automatically
        self.char_to_num = {char: idx for idx, char in enumerate(self.characters)}
        
        # Create index to character mapping
        self.num_to_char = {idx: char for char, idx in self.char_to_num.items()}
        
        # Vocab size includes all characters + blank token at the end
        self.vocab_size = len(self.characters) + 1  # +1 for CTC blank
        self.blank_token_idx = len(self.characters)  # Blank is last index
    
    def encode(self, text):
        """Encode text to numerical indices (0 to len(characters)-1)"""
        encoded = []
        for char in text:
            if char in self.char_to_num:
                encoded.append(self.char_to_num[char])
            # Skip unknown characters instead of using blank
        return encoded
    
    def decode(self, indices):
        """Decode numerical indices to text (skip blank token)"""
        decoded = []
        for idx in indices:
            if idx < len(self.characters) and idx in self.num_to_char:
                decoded.append(self.num_to_char[idx])
            # Skip blank token (self.blank_token_idx) and unknown indices
        return ''.join(decoded)


# ============================================================================
# 3. CNN-LSTM MODEL ARCHITECTURE
# ============================================================================

def build_crnn_model(input_shape=(32, 128, 1), num_classes=79):
    """
    Build CRNN model with 6 Conv layers and 2 BiLSTM layers
    Architecture from the paper
    """
    
    # Input layer
    input_layer = layers.Input(shape=input_shape, name='input_1')
    
    # Convolutional Block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d')(input_layer)
    x = layers.MaxPooling2D((2, 2), name='max_pooling2d')(x)
    
    # Convolutional Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_1')(x)
    x = layers.MaxPooling2D((2, 2), name='max_pooling2d_1')(x)
    
    # Convolutional Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_2')(x)
    
    # Convolutional Block 4
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_3')(x)
    x = layers.MaxPooling2D((2, 1), name='max_pooling2d_2')(x)
    
    # Convolutional Block 5
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv2d_4')(x)
    x = layers.BatchNormalization(name='batch_normalization')(x)
    
    # Convolutional Block 6
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv2d_5')(x)
    x = layers.BatchNormalization(name='batch_normalization_1')(x)
    x = layers.MaxPooling2D((2, 1), name='max_pooling2d_3')(x)
    
    # Convolutional Block 7
    x = layers.Conv2D(512, (2, 2), activation='relu', name='conv2d_6')(x)
    
    # Reshape for LSTM (Lambda layer)
    # Shape: (batch, height, width, channels) -> (batch, width, height*channels)
    x = layers.Lambda(lambda x: tf.squeeze(x, axis=1), name='lambda')(x)
    
    # Bidirectional LSTM layers
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.2), 
                            name='bidirectional')(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.2), 
                            name='bidirectional_1')(x)
    
    # Dense output layer
    output = layers.Dense(num_classes, activation='softmax', name='dense')(x)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output, name='CRNN_HTR')
    
    return model


# ============================================================================
# 4. CTC LOSS FUNCTION
# ============================================================================

def ctc_loss_function(y_true, y_pred):
    """CTC (Connectionist Temporal Classification) loss"""
    # Get batch size
    batch_size = tf.shape(y_true)[0]
    
    # Input length (time steps from model output)
    input_length = tf.shape(y_pred)[1] * tf.ones(shape=(batch_size, 1), dtype='int32')
    
    # Label length (actual label sequence length)
    label_length = tf.reduce_sum(tf.cast(y_true != 0, tf.int32), axis=-1, keepdims=True)
    
    # Calculate CTC loss
    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    
    return loss


# ============================================================================
# 5. DATA GENERATOR
# ============================================================================

class HTRDataGenerator(keras.utils.Sequence):
    """Data generator for HTR training"""
    
    def __init__(self, image_paths, labels, encoder, batch_size=5, 
                 img_height=32, img_width=128, shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.encoder = encoder
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_paths))
        self.on_epoch_end()
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.floor(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Generate data
        X, y = self.__data_generation(indexes)
        
        return X, y
    
    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, indexes):
        """Generate data for a batch"""
        # Initialize arrays
        X = np.zeros((self.batch_size, self.img_height, self.img_width, 1), dtype=np.float32)
        y = np.zeros((self.batch_size, 64), dtype=np.int32)  # Max label length = 64
        
        # Generate data
        for i, idx in enumerate(indexes):
            # Load and preprocess image
            img = preprocess_image(self.image_paths[idx], self.img_height, self.img_width)
            X[i] = img
            
            # Encode label
            encoded_label = self.encoder.encode(self.labels[idx])
            y[i, :len(encoded_label)] = encoded_label
        
        return X, y


# ============================================================================
# 6. JARO-WINKLER SIMILARITY
# ============================================================================

def jaro_winkler_similarity(str1, str2):
    """Calculate Jaro-Winkler similarity between two strings"""
    
    def jaro_similarity(s1, s2):
        if len(s1) == 0 and len(s2) == 0:
            return 1.0
        if len(s1) == 0 or len(s2) == 0:
            return 0.0
        
        match_distance = max(len(s1), len(s2)) // 2 - 1
        s1_matches = [False] * len(s1)
        s2_matches = [False] * len(s2)
        
        matches = 0
        transpositions = 0
        
        for i in range(len(s1)):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len(s2))
            
            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break
        
        if matches == 0:
            return 0.0
        
        k = 0
        for i in range(len(s1)):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1
        
        return (matches / len(s1) + matches / len(s2) + 
                (matches - transpositions / 2) / matches) / 3
    
    jaro_sim = jaro_similarity(str1, str2)
    
    # Calculate prefix length (max 4)
    prefix = 0
    for i in range(min(len(str1), len(str2), 4)):
        if str1[i] == str2[i]:
            prefix += 1
        else:
            break
    
    # Calculate Jaro-Winkler similarity
    jaro_winkler = jaro_sim + (prefix * 0.1 * (1 - jaro_sim))
    
    return jaro_winkler


# ============================================================================
# 7. TRAINING FUNCTION
# ============================================================================

def train_model(model, train_generator, val_generator, epochs=50, 
                checkpoint_path='best_model.h5'):
    """Train the CRNN model"""
    
    # Compile model with SGD optimizer
    optimizer = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=optimizer, loss=ctc_loss_function)
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    
    return history


# ============================================================================
# 8. PREDICTION AND DECODING
# ============================================================================

def decode_predictions(predictions, encoder, blank_index=None):
    """Decode CTC predictions to text
    
    Args:
        predictions: numpy array of shape (batch_size, time_steps, num_classes)
        encoder: CharacterEncoder instance
        blank_index: Index of blank token (optional, uses encoder.blank_token_idx)
    
    Returns:
        List of decoded text strings
    """
    if blank_index is None:
        blank_index = encoder.blank_token_idx
    
    decoded_texts = []
    
    # predictions shape: (batch_size, time_steps, num_classes)
    batch_size = predictions.shape[0]
    time_steps = predictions.shape[1]
    
    # Create input_length for all samples in batch
    input_lengths = np.full((batch_size,), time_steps, dtype=np.int32)
    
    # Decode all predictions at once
    decoded, _ = tf.keras.backend.ctc_decode(
        predictions,
        input_length=input_lengths,
        greedy=True
    )
    
    # Convert to text
    decoded = decoded[0].numpy()
    for i in range(batch_size):
        # Get the decoded sequence for this sample
        seq = decoded[i]
        # Decode to text using encoder
        text = encoder.decode(seq)
        decoded_texts.append(text)
    
    return decoded_texts


# ============================================================================
# 9. VISUALIZATION FUNCTIONS
# ============================================================================

def plot_training_history(history, epochs_trained):
    """Plot training and validation accuracy/loss"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training and Validation Loss ({epochs_trained} Epochs)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy (if available)
    if 'accuracy' in history.history:
        ax2.plot(history.history['accuracy'], label='Training Accuracy')
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'Training and Validation Accuracy ({epochs_trained} Epochs)')
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'training_history_{epochs_trained}_epochs.png')
    plt.show()


def visualize_predictions(image_paths, true_labels, predicted_labels, num_samples=5):
    """Visualize predictions with original images"""
    
    num_samples = min(num_samples, len(image_paths))
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, num_samples * 2))
    
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        # Load and display image
        img = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        
        # Calculate similarity
        similarity = jaro_winkler_similarity(true_labels[i], predicted_labels[i])
        
        # Set title with true and predicted text
        title = f"True: {true_labels[i]}\nPredicted: {predicted_labels[i]}\nSimilarity: {similarity:.2%}"
        axes[i].set_title(title, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('prediction_samples.png')
    plt.show()


# ============================================================================
# 10. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("=" * 80)
    print("Handwritten Text Recognition Using CNN-LSTM")
    print("=" * 80)
    
    # Initialize character encoder
    print("\n[1/7] Initializing character encoder...")
    encoder = CharacterEncoder()
    print(f"Vocabulary size: {encoder.vocab_size}")
    
    # Build model
    print("\n[2/7] Building CRNN model...")
    model = build_crnn_model(input_shape=(32, 128, 1), num_classes=encoder.vocab_size)
    model.summary()
    
    # Print model statistics
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\n[3/7] Model architecture successfully created!")
    print("\nNote: To train the model, you need:")
    print("  - IAM Dataset with image paths and labels")
    print("  - Create train/validation data generators")
    print("  - Call train_model() function")
    
    print("\n[4/7] Example usage:")
    print("""
    # Example: Load your dataset
    train_images = ['path/to/image1.png', 'path/to/image2.png', ...]
    train_labels = ['hello', 'world', ...]
    
    # Create data generator
    train_gen = HTRDataGenerator(train_images, train_labels, encoder, batch_size=5)
    val_gen = HTRDataGenerator(val_images, val_labels, encoder, batch_size=5)
    
    # Train model
    history = train_model(model, train_gen, val_gen, epochs=50)
    
    # Evaluate
    predictions = model.predict(test_images)
    decoded = decode_predictions(predictions, encoder)
    """)
    
    print("\n[5/7] Preprocessing example:")
    # Create a sample image for demonstration
    sample_img = np.ones((50, 200, 3), dtype=np.uint8) * 255
    cv2.putText(sample_img, "SAMPLE", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imwrite('sample_input.png', sample_img)
    
    # Preprocess it
    preprocessed = preprocess_image('sample_input.png')
    print(f"Preprocessed shape: {preprocessed.shape}")
    print(f"Value range: [{preprocessed.min():.2f}, {preprocessed.max():.2f}]")
    
    print("\n[6/7] Character encoding example:")
    sample_text = "Hello World"
    encoded = encoder.encode(sample_text)
    decoded = encoder.decode(encoded)
    print(f"Original: {sample_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    print("\n[7/7] Jaro-Winkler similarity example:")
    text1 = "recognition"
    text2 = "recogniton"
    similarity = jaro_winkler_similarity(text1, text2)
    print(f"'{text1}' vs '{text2}': {similarity:.2%}")
    
    print("\n" + "=" * 80)
    print("Setup complete! Model ready for training.")
    print("=" * 80)
    
    return model, encoder


if __name__ == "__main__":
    model, encoder = main()
