# Language Translation Transformers

![image](https://github.com/Dharinesh/Transformers-for-language-translation/assets/108059896/e92e7202-b14b-447e-b519-c9de1881222c)


## Introduction

This project implements a Transformer-based neural machine translation model for English to Hindi translation. The model is built from scratch using PyTorch, following the architecture described in the paper "Attention Is All You Need" by Vaswani et al.

## What are Transformers?

Transformers are a type of neural network architecture that has revolutionized natural language processing tasks. Unlike traditional sequential models, Transformers use self-attention mechanisms to process input sequences in parallel, allowing them to capture long-range dependencies more effectively.

Key components of a Transformer include:

1. Embedding layers
2. Positional encoding
3. Multi-head attention mechanisms
4. Feed-forward neural networks
5. Layer normalization and residual connections

## How Transformers Work

Transformers operate through a series of sophisticated processing steps:

1. Input Embedding: The input sentence is converted into numerical vectors (embeddings). Each word or subword is represented as a high-dimensional vector.

2. Positional Encoding: Since Transformers process all input tokens in parallel, positional information is added to the embeddings. This allows the model to understand the sequence order.

3. Encoder:
   - The encoder consists of multiple identical layers.
   - Each layer has two sub-layers:
     a. Multi-Head Self-Attention: This mechanism allows the model to focus on different parts of the input sequence when encoding each word. It helps in capturing contextual relationships.
     b. Feed-Forward Neural Network: A simple fully connected network applied to each position separately and identically.
   - Each sub-layer is followed by layer normalization and is wrapped with a residual connection.

4. Decoder:
   - The decoder also consists of multiple identical layers.
   - Each layer has three sub-layers:
     a. Masked Multi-Head Self-Attention: Similar to the encoder, but masks future positions to prevent the model from looking ahead.
     b. Multi-Head Attention: This attends to the encoder's output, allowing the decoder to focus on relevant parts of the input sequence.
     c. Feed-Forward Neural Network: Similar to the encoder.
   - Layer normalization and residual connections are used throughout.

5. Output Generation:
   - The decoder generates the output sequence one token at a time.
   - At each step, the previously generated tokens are fed back into the decoder.
   - The final linear layer and softmax function convert the decoder's output into probabilities over the target vocabulary.

This architecture allows Transformers to efficiently process sequences, capturing both local and global dependencies, making them highly effective for tasks like translation.

## This Project

This project implements a Transformer model for English to Hindi translation. Here's a detailed breakdown of the implementation:

1. Data Preparation:
   - The project uses the "English-Hindi-Translation" dataset from Hugging Face.
   - A custom `TranslationDataset` class is implemented to handle data loading and preprocessing.
   - Vocabularies are built for both English and Hindi languages.

2. Model Architecture:
   - The `Transformer` class implements the core model architecture.
   - It includes custom embedding layers, positional encoding, and uses PyTorch's `nn.Transformer` for the main transformer blocks.
   - A final linear layer is used to project the decoder output to the target vocabulary size.

3. Training Process:
   - The model is trained using CrossEntropyLoss and the Adam optimizer.
   - Gradient clipping is applied to prevent exploding gradients.
   - The training loop processes batches of data, computes loss, and updates model parameters.

4. Translation Function:
   - A `translate` function is implemented to perform inference on new English sentences.
   - It handles the step-by-step decoding process, generating Hindi translations.

5. Model Saving and Loading:
   - The trained model, along with vocabularies, is saved to disk.
   - A custom `HFTransformer` class is created to make the model compatible with the Hugging Face ecosystem.

6. Tokenizer Integration:
   - Custom tokenizers are created for both source and target languages.
   - These are saved in a format compatible with Hugging Face's `PreTrainedTokenizer`.

7. Hugging Face Integration:
   - The model, tokenizers, and configuration are uploaded to the Hugging Face Model Hub.
   - This allows easy sharing and deployment of the trained model.

## Model on Hugging Face

The trained model is available on the Hugging Face Model Hub. You can access and use it here:

[Transformer-English-to-Hindi-translation](https://huggingface.co/Dharinesh/Transformer-English-to-Hindi-translation)

## Results

- Training progression :
  ![image](https://github.com/Dharinesh/Transformers-for-language-translation/assets/108059896/f9e4b778-1a27-44db-9a12-91b52a01c4c3)

- Some Results:
  ![image](https://github.com/Dharinesh/Transformers-for-language-translation/assets/108059896/3624e42e-6623-4b65-ab91-f74743f01126)
  ![image](https://github.com/Dharinesh/Transformers-for-language-translation/assets/108059896/d228c171-65a1-4f0d-a286-8954aca11918)
  ![image](https://github.com/Dharinesh/Transformers-for-language-translation/assets/108059896/abe35cc2-b0ca-46f8-a7bd-75bd83517007)
  ![image](https://github.com/Dharinesh/Transformers-for-language-translation/assets/108059896/3bc9e676-fe5e-471d-8f37-a553cbebc2c2)


## References

- Vaswani, A., et al. (2017). "Attention Is All You Need". In Advances in Neural Information Processing Systems (NIPS 2017).

## License

- This is using GNU General License ... feel free to use the resources
