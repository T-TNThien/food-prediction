Problem: Recognizing food names from images
Applied model: Transformer Encoder – Decoder
# Overview of Transformer architecture
Transformer is an Attention-based model, introduced by Vaswani et al. (2017). Transformer consists of two main components: Encoder and Decoder.
## Encoder
Encoder consists of N layers, each layer has the following structure:
• Self-Attention Mechanism: Calculates the importance of each element in the input sequence compared to all other elements. The formula for calculating Attention is shown as follows:

![image](https://github.com/user-attachments/assets/b222e571-8902-4c15-8421-3483f524ce9c)

• Feed-Forward Neural Network (FFNN): Two-layer linear neural network, applied independently to each position:

![image](https://github.com/user-attachments/assets/3fc12e67-3e2c-4481-8582-29d7100be10b)

• Add & Norm: After each Self-Attention and FFNN layer is an addition and normalization step (Layer Normalization).
## Decoder
Decoder also consists of N layers similar to Encoder, but has an additional Attention layer to focus on the output of Encoder.
• Masked Self-Attention: Similar to Self-Attention, but with a mask to suppress information from future positions in the sequence.
• Encoder-Decoder Attention: Calculates the Attention between the Encoder output and the Decoder's current input.
## Dataset
The dataset consists of 16,643 food images divided into 11 classes. The data is divided into training, validation, and test sets.
Preprocess the data:
• Resize the images to a fixed size (e.g. 224x224 pixels).
• Normalize the pixel values.
## Model Architecture
Using Vision Transformer (ViT) for food recognition:
• Patch Embedding: Divide the image into patches and convert each patch into an embedding vector. With a 224x224 image and a patch size of 16x16, we have 196 patches
• Positional Encoding: Adds positional information to each patch to retain image structure information.
• Transformer Layers: Apply N Encoder layers of the Transformer to the patch embeddings that have been added positional encoding.
• Classification Head: Uses the output of the last Transformer layer to predict the class of the image through a fully connected layer.
## Model Training
Using the prepared training data set, train the model with the Cross-Entropy loss function and optimizer like Adam.
