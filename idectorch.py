# Pytorch implementation of IDEC Algorithm of dimensionality reduction + clustering using autoencoders.


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import  Dataset
import numpy as np
from sklearn.cluster import KMeans


class UnsupervisedDataset(Dataset):
    """Dataset for unsupervised learning - only returns data, no labels"""

    def __init__(self, data):
        self.data = torch.FloatTensor(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class IDEC(nn.Module):
    def __init__(self, input_dim=784, z_dim=10, n_clusters=10,
                 encodeLayer=[400], decodeLayer=[400], activation="relu", dropout=0):
        super(IDEC, self).__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.n_clusters = n_clusters
        self.activation = activation
        self.dropout = dropout

        # Activation function
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "leakyrelu":
            self.act = nn.LeakyReLU(0.1)

        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for dim in encodeLayer:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                self.act
            ])
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            prev_dim = dim

        # Add final encoder layer (to latent space)
        encoder_layers.append(nn.Linear(prev_dim, z_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder layers
        decoder_layers = []
        prev_dim = z_dim
        for dim in decodeLayer:
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                self.act
            ])
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            prev_dim = dim

        # Add final decoder layer (back to input space)
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Clustering layer
        self.cluster_layer = ClusteringLayer(n_clusters, z_dim)

    def forward(self, x):
        # Encode
        z = self.encoder(x)

        # Decode
        x_reconstructed = self.decoder(z)

        # Cluster
        q = self.cluster_layer(z)

        return x_reconstructed, q, z

    def encode(self, x):
        """Get latent representation only"""
        return self.encoder(x)


class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters, z_dim, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.weight = nn.Parameter(torch.Tensor(n_clusters, z_dim))
        nn.init.xavier_normal_(self.weight)

    def forward(self, z):
        # Compute soft assignment using Student's t-distribution
        # q_ij = (1 + ||z_i - μ_j||²/α)^(-(α+1)/2) / Σ_j'(1 + ||z_i - μ_j'||²/α)^(-(α+1)/2)

        # Expand dimensions for broadcasting
        z_expanded = z.unsqueeze(1)  # (batch_size, 1, z_dim)
        weight_expanded = self.weight.unsqueeze(0)  # (1, n_clusters, z_dim)

        # Compute squared distances
        distances = torch.sum((z_expanded - weight_expanded) ** 2, dim=2)  # (batch_size, n_clusters)

        # Compute soft assignments
        numerator = (1.0 + distances / self.alpha) ** (-(self.alpha + 1.0) / 2.0)
        denominator = torch.sum(numerator, dim=1, keepdim=True)

        q = numerator / denominator
        return q


def target_distribution(q, power=2.0, temperature=1.0):
    """
    Compute target distribution P using current soft assignments Q
    P_ij = (q_ij^power/f_j) / Σ_j'(q_ij'^power/f_j')
    where f_j = Σ_i q_ij (cluster frequencies)

    Args:
        power: exponent for sharpening (lower = less sharp, try 1.5 or 1.0)
        temperature: softening factor (higher = softer, try 2.0 or 3.0)
    """
    weight = (q ** power / q.sum(0)) ** (1.0 / temperature)
    return (weight.t() / weight.sum(1)).t()


class IDECTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device

    def pretrain(self, dataloader, lr=0.001, epochs=200):
        """
        Pretrain the autoencoder
        """
        print("=== Pretraining Autoencoder ===")
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, data in enumerate(dataloader):
                data = data.view(data.size(0), -1).to(self.device)

                optimizer.zero_grad()

                # Forward pass
                x_reconstructed, _, _ = self.model(data)

                # Reconstruction loss
                loss = F.mse_loss(x_reconstructed, data)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 50 == 0:
                print(f'Pretraining Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.6f}')

    def initialize_clusters(self, dataloader):
        """
        Initialize cluster centers using K-means on the encoded features
        """
        print("=== Initializing Clusters ===")
        self.model.eval()

        # Get all encoded features
        encoded_features = []
        with torch.no_grad():
            for data in dataloader:
                data = data.view(data.size(0), -1).to(self.device)
                _, _, z = self.model(data)
                encoded_features.append(z.cpu().numpy())

        encoded_features = np.concatenate(encoded_features, axis=0)

        # Perform K-means
        kmeans = KMeans(n_clusters=self.model.n_clusters, random_state=42, n_init=20)
        cluster_labels = kmeans.fit_predict(encoded_features)

        # Set cluster centers
        self.model.cluster_layer.weight.data = torch.tensor(kmeans.cluster_centers_).to(self.device)

        return kmeans.labels_

    def train(self, dataloader, lr=0.001, epochs=100, update_interval=10,
              tol=1e-3, gamma=0.1, td_power = 2.0, td_temp = 1.0):
        """
        Train IDEC with joint optimization
        Args:
            gamma: weight for reconstruction loss (reconstruction_loss_weight)
        """
        print("=== Training IDEC ===")

        # Initialize clusters
        y_pred_last = self.initialize_clusters(dataloader)

        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            total_cluster_loss = 0
            total_recon_loss = 0

            # Update target distribution every update_interval epochs
            if epoch % update_interval == 0:
                # Get current cluster assignments
                q_all = []
                self.model.eval()
                with torch.no_grad():
                    for data in dataloader:
                        data = data.view(data.size(0), -1).to(self.device)
                        _, q, _ = self.model(data)
                        q_all.append(q.cpu())

                q_all = torch.cat(q_all, dim=0)
                p_all = target_distribution(q_all, power=td_power, temperature=td_temp).to(self.device)

                # Check for convergence
                y_pred = q_all.argmax(1).numpy()
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred

                print(f'Epoch {epoch}, delta_label: {delta_label:.4f}')

                if epoch > 0 and delta_label < tol:
                    print(f'Converged at epoch {epoch}')
                    break

            self.model.train()

            batch_start = 0
            for batch_idx, data in enumerate(dataloader):
                data = data.view(data.size(0), -1).to(self.device)
                batch_size = data.size(0)

                optimizer.zero_grad()

                # Forward pass
                x_reconstructed, q, z = self.model(data)

                # Get corresponding target distribution batch
                if epoch % update_interval == 0 or epoch == 0:
                    p_batch = p_all[batch_start:batch_start + batch_size]
                    batch_start += batch_size
                else:
                    # Use current q to compute p for this batch
                    p_batch = target_distribution(q)

                # Clustering loss (KL divergence)
                cluster_loss = F.kl_div(q.log(), p_batch, reduction='batchmean')

                # Reconstruction loss
                recon_loss = F.mse_loss(x_reconstructed, data)

                # Total loss
                loss = cluster_loss + gamma * recon_loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_cluster_loss += cluster_loss.item()
                total_recon_loss += recon_loss.item()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{epochs}:')
                print(f'  Total Loss: {total_loss / len(dataloader):.6f}')
                print(f'  Cluster Loss: {total_cluster_loss / len(dataloader):.6f}')
                print(f'  Recon Loss: {total_recon_loss / len(dataloader):.6f}')

    def predict(self, dataloader):
        """
        Get cluster predictions (hard assignments)
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for data in dataloader:
                data = data.view(data.size(0), -1).to(self.device)
                _, q, _ = self.model(data)
                pred = q.argmax(1)
                predictions.append(pred.cpu().numpy())

        return np.concatenate(predictions)

    def get_latent_representation(self, dataloader):
        """
        Get latent space representations (embeddings)
        """
        self.model.eval()
        latent_features = []

        with torch.no_grad():
            for data in dataloader:
                data = data.view(data.size(0), -1).to(self.device)
                z = self.model.encode(data)
                latent_features.append(z.cpu().numpy())

        return np.concatenate(latent_features, axis=0)

    def get_soft_assignments(self, dataloader):
        """
        Get soft cluster assignments (probabilities)
        """
        self.model.eval()
        soft_assignments = []

        with torch.no_grad():
            for data in dataloader:
                data = data.view(data.size(0), -1).to(self.device)
                _, q, _ = self.model(data)
                soft_assignments.append(q.cpu().numpy())

        return np.concatenate(soft_assignments, axis=0)

    def reconstruct(self, dataloader):
        """
        Get reconstructed data
        """
        self.model.eval()
        reconstructions = []

        with torch.no_grad():
            for data in dataloader:
                data = data.view(data.size(0), -1).to(self.device)
                x_reconstructed, _, _ = self.model(data)
                reconstructions.append(x_reconstructed.cpu().numpy())

        return np.concatenate(reconstructions, axis=0)