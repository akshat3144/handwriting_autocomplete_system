# 🏗️ Phase 1 Architecture: Optical Character Recognition (OCR)

This document details the architecture and implementation of the OCR module, which is responsible for segmenting handwritten sentences into words and recognizing the text using a Deep Learning model.

## 🧠 System Overview

The OCR pipeline transforms a raw image of a handwritten sentence into digital text through a multi-stage process:

1.  **Word Segmentation**: Computer Vision techniques segment the sentence into individual word images.
2.  **Preprocessing**: Each word image is normalized to a fixed size and format.
3.  **Recognition**: A CRNN (Convolutional Recurrent Neural Network) predicts the character sequence.
4.  **Decoding**: CTC (Connectionist Temporal Classification) decoding converts model outputs to text.

---

## ✂️ Word Segmentation Module

The segmentation logic is encapsulated in the `WordSegmenter` class within `segmentation/segmenter.py`. It uses a **7-step Computer Vision pipeline** to isolate words without using deep learning, ensuring speed and robustness.

### Pipeline Steps

1.  **Gaussian Blur**: Applies a `(3, 3)` kernel to reduce high-frequency noise while preserving edges.
2.  **Sobel Edge Detection**: Computes gradients in both X and Y directions for each RGB channel and combines them to find strong edges.
3.  **Binary Thresholding**: Converts the edge map to a binary image using a fixed threshold (50/255).
4.  **Morphological Closing**: Uses a `(3, 3)` kernel to close small gaps within letters.
5.  **Dilation**: Applies a `(1, 3)` kernel (vertical dilation) to connect disjoint parts of characters within a word without merging adjacent words.
6.  **Contour Detection**: Finds external contours in the processed image.
7.  **Filtering & Sorting**:
    *   **Filtering**: Removes noise based on minimum size (`15x10` px), aspect ratio, and fill ratio.
    *   **Sorting**: Groups bounding boxes into lines based on Y-coordinates and then sorts them left-to-right.

---

## 🔄 Preprocessing Pipeline

Before being fed into the neural network, each segmented word image undergoes a standardized preprocessing routine defined in `training/htr_cnn_lstm.py`.

1.  **Grayscale Conversion**: Converts RGB images to grayscale using standard luminosity weights.
2.  **Binarization**: Applies **Otsu's Thresholding** to separate ink from background dynamically.
3.  **Resize & Pad**:
    *   Target Dimensions: **32 (Height) × 128 (Width)**.
    *   Preserves aspect ratio by resizing the image until one dimension fits the target.
    *   Pads the remaining area with white pixels (255) to center the word.
4.  **Normalization**: Scales pixel values from `[0, 255]` to `[0.0, 1.0]`.

---

## 🤖 Model Architecture: CRNN

The core recognition engine is a **Convolutional Recurrent Neural Network (CRNN)**, combining CNNs for feature extraction and LSTMs for sequence modeling.

### 1. Convolutional Feature Extractor (CNN)
The model uses 7 convolutional blocks to extract visual features from the input image `(32, 128, 1)`.

| Layer | Type | Filters | Kernel | Stride | Activation | Output Shape |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Input** | Input | - | - | - | - | `(32, 128, 1)` |
| **Conv1** | Conv2D | 64 | 3x3 | 1 | ReLU | `(32, 128, 64)` |
| **Pool1** | MaxPool | - | 2x2 | 2 | - | `(16, 64, 64)` |
| **Conv2** | Conv2D | 128 | 3x3 | 1 | ReLU | `(16, 64, 128)` |
| **Pool2** | MaxPool | - | 2x2 | 2 | - | `(8, 32, 128)` |
| **Conv3** | Conv2D | 256 | 3x3 | 1 | ReLU | `(8, 32, 256)` |
| **Conv4** | Conv2D | 256 | 3x3 | 1 | ReLU | `(8, 32, 256)` |
| **Pool3** | MaxPool | - | 2x1 | (2, 1) | - | `(4, 32, 256)` |
| **Conv5** | Conv2D | 512 | 3x3 | 1 | ReLU | `(4, 32, 512)` |
| **BN1** | BatchNorm | - | - | - | - | `(4, 32, 512)` |
| **Conv6** | Conv2D | 512 | 3x3 | 1 | ReLU | `(4, 32, 512)` |
| **BN2** | BatchNorm | - | - | - | - | `(4, 32, 512)` |
| **Pool4** | MaxPool | - | 2x1 | (2, 1) | - | `(2, 32, 512)` |
| **Conv7** | Conv2D | 512 | 2x2 | 1 | ReLU | `(1, 31, 512)` |

### 2. Map-to-Sequence
A `Lambda` layer squeezes the height dimension to prepare the data for the RNN.
*   **Operation**: `squeeze(axis=1)`
*   **Transformation**: `(Batch, 1, 31, 512)` ➝ `(Batch, 31, 512)`
*   This results in a sequence of 31 time steps, each with a 512-dimensional feature vector.

### 3. Recurrent Layers (RNN)
Two Bidirectional LSTM layers capture context from both directions (left-to-right and right-to-left).

*   **BiLSTM 1**: 256 units (returns sequences), Dropout 0.2. Output: `(Batch, 31, 512)`
*   **BiLSTM 2**: 256 units (returns sequences), Dropout 0.2. Output: `(Batch, 31, 512)`

### 4. Output Layer
*   **Dense**: Maps the LSTM output to the character set size.
*   **Activation**: Softmax.
*   **Output Shape**: `(Batch, 31, Num_Classes)` (where Num_Classes ≈ 80, including blank).

---

## 🎓 Training Strategy

The training logic is handled in `training/train_htr_iam.py`.

### 1. Loss Function: CTC
**Connectionist Temporal Classification (CTC)** is used to train the model without requiring alignment between input image pixels and output characters.
*   The model outputs a probability distribution over characters for each time step.
*   CTC Loss calculates the probability of the target sequence given these distributions, summing over all possible alignments.

### 2. Optimization
*   **Optimizer**: `Adam`
*   **Learning Rate**: `0.001`
*   **Callbacks**:
    *   `ModelCheckpoint`: Saves the best model based on validation loss.
    *   `EarlyStopping`: Stops training if validation loss doesn't improve for a set number of epochs.
    *   `ReduceLROnPlateau`: Reduces learning rate when learning stagnates.

### 3. Data Handling
*   **Dataset**: IAM Handwriting Database.
*   **Generator**: `CTCDataGenerator` (subclass of `keras.utils.Sequence`) handles batching, shuffling, and on-the-fly preprocessing.

---

## 🔮 Inference Pipeline

The `sentence_recognizer.py` script orchestrates the inference process.

### 1. Character Encoding
A `CharacterEncoder` class manages the mapping between characters and numerical indices.
*   **Vocabulary**: `abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'-` + `[blank]`

### 2. Decoding
The raw output from the model is a matrix of probabilities `(31, Num_Classes)`.
*   **CTC Greedy Decoder**: Selects the character with the highest probability at each time step.
*   **Collapse**: Merges repeated characters and removes blank tokens to form the final string.

### 3. Execution Flow
1.  Load Image.
2.  `WordSegmenter.segment()` ➝ List of word images.
3.  For each word image:
    *   `preprocess_image()` ➝ `(32, 128, 1)` tensor.
    *   `model.predict()` ➝ Logits.
    *   `decoder` ➝ Text string.
4.  Concatenate words to form the final sentence.

---

## 📂 File Structure Summary

| File | Description |
| :--- | :--- |
| `sentence_recognizer.py` | Main entry point for end-to-end sentence recognition. |
| `segmentation/segmenter.py` | Computer vision logic for word segmentation. |
| `training/htr_cnn_lstm.py` | Definition of the CRNN model and preprocessing functions. |
| `training/train_htr_iam.py` | Script for training the model on the IAM dataset. |
| `model_weights/` | Directory containing trained `.h5` models and `.pkl` encoders. |
