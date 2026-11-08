import torch
import torch.nn as nn

class FTTransformerEncoder(nn.Module):
    def __init__(self, num_features=5, token_dim=64, num_layers=2, nhead=4, embed_dim=256):
        """
        FT-Transformer architecture for tabular data.
        num_features: Number of clinical variables (default is 5 for our BRSET setup)
        token_dim: Hidden dimension for each feature token
        embed_dim: Final output dimension to match the Vision Encoder (256)
        """
        super(FTTransformerEncoder, self).__init__()
        self.num_features = num_features
        
        # 1. Feature Tokenizer (Projects scalars to vectors)
        self.feature_embeddings = nn.Parameter(torch.randn(num_features, token_dim))
        self.feature_biases = nn.Parameter(torch.randn(num_features, token_dim))
        
        # CLS Token (Used to aggregate the global tabular representation)
        self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim))
        
        # 2. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim, 
            nhead=nhead, 
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Projection Head (Matches the contrastive space of the Vision Encoder)
        self.projection_head = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.BatchNorm1d(token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, embed_dim)
        )
        
    def forward(self, x, mask):
        """
        x: Tabular numerical features [Batch, num_features]
        mask: Missing value mask [Batch, num_features] (1=valid, 0=missing/NaN)
        """
        B, N = x.shape
        
        # Step 1: Tokenize features
        # x shape changes from (B, N) to (B, N, token_dim)
        tokens = x.unsqueeze(-1) * self.feature_embeddings.unsqueeze(0) + self.feature_biases.unsqueeze(0)
        
        # Step 2: Apply the mask (Zero out tokens that represent missing/NaN data)
        tokens = tokens * mask.unsqueeze(-1)
        
        # Step 3: Append the CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1) # Shape: (B, N+1, token_dim)
        
        # Step 4: Create Attention Padding Mask for the Transformer
        # In PyTorch, 'True' means the position should be IGNORED.
        # Since our mask uses 0 for missing, (mask == 0) gives 'True' for missing features.
        padding_mask = (mask == 0)
        
        # The CLS token is at index 0 and should never be ignored (False)
        cls_padding = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        key_padding_mask = torch.cat((cls_padding, padding_mask), dim=1)
        
        # Step 5: Pass through Transformer
        transformed_tokens = self.transformer(tokens, src_key_padding_mask=key_padding_mask)
        
        # Step 6: Extract the aggregated representation from the CLS token
        cls_output = transformed_tokens[:, 0, :]
        
        # Step 7: Project to common embedding space and apply L2 normalization
        embeddings = self.projection_head(cls_output)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings

# ==========================================
# Local testing block
# ==========================================
if __name__ == "__main__":
    batch_size = 4
    num_features = 5
    
    # Simulate a batch of tabular data and masks
    dummy_tabular = torch.randn(batch_size, num_features)
    
    # Simulate a mask (e.g., patient 2 has missing data in the 3rd feature)
    dummy_mask = torch.ones(batch_size, num_features)
    dummy_mask[1, 2] = 0.0 
    
    model = FTTransformerEncoder(num_features=num_features, embed_dim=256)
    
    output_embeddings = model(dummy_tabular, dummy_mask)
    
    print("Tabular Input Shape:", dummy_tabular.shape)
    print("Mask Input Shape:", dummy_mask.shape)
    print("Tabular Output Embedding (T):", output_embeddings.shape)