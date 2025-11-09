from common_imports import *

class DiffusionModel(nn.Module):
    def __init__(self, T_diffusion=200, time_embed_dim=32, hidden_dim=64):
        super(DiffusionModel, self).__init__()
        self.time_embed = nn.Embedding(T_diffusion, time_embed_dim)
        self.conv1 = nn.Conv1d(1 + time_embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        t_emb = self.time_embed(t)
        t_emb = t_emb.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat([x, t_emb], dim=1)
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        out = self.conv3(h)
        return out.squeeze(1)
    
class LatentDiffusionModel(nn.Module):
    def __init__(self, T_diffusion=200, time_embed_dim=64, hidden_dim=128, latent_dim=64, num_heads=4):
        super(LatentDiffusionModel, self).__init__()
        self.T_diffusion = T_diffusion
        self.latent_dim = latent_dim
        self.time_embed_dim = time_embed_dim

        # Time embedding with MLP
        self.time_embed = nn.Embedding(T_diffusion, time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # Encoder: Input (1 channel) -> Latent space (latent_dim channels) with residual connections
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1 if i == 0 else hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ) for i in range(2)
        ])
        self.encoder_out = nn.Conv1d(hidden_dim, latent_dim, kernel_size=3, padding=1)

        # Diffusion network in latent space with self-attention
        self.attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, batch_first=True)
        self.diffusion_net = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(latent_dim + time_embed_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
        ])
        self.diffusion_out = nn.Conv1d(hidden_dim, latent_dim, kernel_size=3, padding=1)

        # Decoder: Latent space -> Output (1 channel) with residual connections
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(latent_dim if i == 0 else hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ) for i in range(2)
        ])
        self.decoder_out = nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)

    def forward(self, x, t):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, seq_len)

        # Time embedding
        t_emb = self.time_embed(t)  # (batch_size, time_embed_dim)
        t_emb = self.time_mlp(t_emb)  # (batch_size, time_embed_dim)
        t_emb = t_emb.unsqueeze(-1).expand(-1, -1, x.size(-1))  # (batch_size, time_embed_dim, seq_len)

        # Encoder with residual connections
        h = x
        for layer in self.encoder:
            residual = h
            h = layer(h)
            if residual.size(1) == h.size(1):  # Channel dimension check
                h = h + residual
        z = self.encoder_out(h)  # (batch_size, latent_dim, seq_len)

        # Diffusion network with self-attention
        z_t = z.permute(0, 2, 1)  # (batch_size, seq_len, latent_dim)
        z_t, _ = self.attention(z_t, z_t, z_t)  # (batch_size, seq_len, latent_dim)
        z_t = z_t.permute(0, 2, 1)  # (batch_size, latent_dim, seq_len)

        # Concatenate with time embedding
        z_t = torch.cat([z_t, t_emb], dim=1)  # (batch_size, latent_dim + time_embed_dim, seq_len)

        # Diffusion process
        h = z_t
        for layer in self.diffusion_net:
            residual = h
            h = layer(h)
            if residual.size(1) == h.size(1):
                h = h + residual
        z_out = self.diffusion_out(h)  # (batch_size, latent_dim, seq_len)

        # Decoder with residual connections
        h = z_out
        for layer in self.decoder:
            residual = h
            h = layer(h)
            if residual.size(1) == h.size(1):
                h = h + residual
        out = self.decoder_out(h)  # (batch_size, 1, seq_len)

        return out.squeeze(1)  # (batch_size, seq_len)