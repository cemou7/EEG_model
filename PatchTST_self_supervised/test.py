import torch
import torch.nn as nn
import torch.optim as optim

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers):
        super(TransformerAutoencoder, self).__init__()
        self.patch_embedding = nn.Linear(input_dim, model_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.linear = nn.Linear(model_dim, input_dim)

    def forward(self, src, mask):
        # Create patch embeddings
        src = self.patch_embedding(src)
        # Apply transformer encoder
        encoded_src = self.transformer_encoder(src, src_key_padding_mask=mask)
        # Reconstruct the masked patches
        reconstructed_src = self.linear(encoded_src)
        return reconstructed_src

# Hyperparameters
input_dim = 128  # Size of one time series data point
model_dim = 512  # Embedding size
num_heads = 8    # Number of heads in the multi-head attention models
num_layers = 3   # Number of transformer layers
sequence_length = 100  # Length of the time series sequence

# Create random data for this example
data = torch.randn(sequence_length, input_dim)

# Masking 15% of the data as suggested by BERT
mask = torch.rand(sequence_length) < 0.15
masked_data = data.clone()
# Replace the masked input values with zero
masked_data[mask] = 0

# Create the model and optimizer
model = TransformerAutoencoder(input_dim, model_dim, num_heads, num_layers)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Forward pass
reconstructed_data = model(masked_data.unsqueeze(0), mask.unsqueeze(0).t())

# Compute the loss only for masked parts
loss_fn = nn.MSELoss(reduction='none')
loss = loss_fn(reconstructed_data.squeeze(0), data)
loss = loss[mask].mean()

# Backpropagation
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Loss: {loss.item()}")
