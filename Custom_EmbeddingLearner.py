import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter
import random

# Define the Skip-gram model
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, target_word):
        # Get the word embedding for the target word
        embedded = self.embedding(target_word)
        
        # Pass the embedding through the output layer
        output = self.output_layer(embedded)
        
        return output

# Prepare the data
def prepare_data(text, vocab_size, context_window):
    # Tokenize the text
    words = text.split()
    
    # Count word frequencies and build vocabulary
    word_counts = Counter(words)
    vocab = {word: idx for idx, (word, _) in enumerate(word_counts.most_common(vocab_size))}
    
    # Create pairs of target and context words
    data = []
    for idx, word in enumerate(words):
        if word in vocab:
            target_idx = vocab[word]
            # Get context words within the specified context window
            context_indices = list(range(max(0, idx - context_window), min(len(words), idx + context_window + 1)))
            context_indices.remove(idx)  # Exclude the target word itself
            
            for context_idx in context_indices:
                context_word = words[context_idx]
                if context_word in vocab:
                    context_idx = vocab[context_word]
                    data.append((target_idx, context_idx))
                    
    return data, vocab

# Train the model
def train_model(text, vocab_size, embedding_dim, context_window, epochs, batch_size, learning_rate):
    # Prepare the data
    data, vocab = prepare_data(text, vocab_size, context_window)
    
    # Create the model
    model = SkipGram(vocab_size, embedding_dim)
    
    # Define loss function (negative sampling) and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(data)
        
        # Batch training
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            
            # Separate batch data into target and context words
            targets, contexts = zip(*batch_data)
            targets = torch.tensor(targets, dtype=torch.long)
            contexts = torch.tensor(contexts, dtype=torch.long)
            
            # Forward pass
            output = model(targets)
            
            # Calculate loss
            loss = loss_fn(output, contexts)
            total_loss += loss.item()
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Print loss for the epoch
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data):.4f}")
    
    # Return the trained model and the vocabulary
    return model, vocab

if __name__ == "__main__":
    # Sample text data
    text = "The quick brown fox jumps over the lazy dog"
    
    # Hyperparameters
    vocab_size = 100  # Maximum size of the vocabulary
    embedding_dim = 50  # Dimension of word embeddings
    context_window = 2  # Context window size
    epochs = 10  # Number of training epochs
    batch_size = 4  # Batch size for training
    learning_rate = 0.001  # Learning rate
    
    # Train the model
    model, vocab = train_model(text, vocab_size, embedding_dim, context_window, epochs, batch_size, learning_rate)
    
    # Example: Get the embedding for a word
    word = "fox"
    if word in vocab:
        word_idx = vocab[word]
        embedding = model.embedding(torch.tensor(word_idx))
        print(f"Embedding for '{word}':")
        print(embedding)
