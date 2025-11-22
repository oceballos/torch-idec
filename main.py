# Example usage

from idectorch import IDEC, IDECTrainer, UnsupervisedDataset
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import torch
from torch.utils.data import DataLoader


# Example usage
if __name__ == "__main__":
    # Generate sample data (you would replace this with your actual data)
    from sklearn.datasets import make_blobs

    # Create sample dataset
    X, y_true = make_blobs(n_samples=1000, centers=4, n_features=20,
                           random_state=42, cluster_std=1.5)

    # Create unsupervised dataset (no labels needed for training)
    dataset = UnsupervisedDataset(X)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = IDEC(input_dim=20, z_dim=10, n_clusters=4,
                 encodeLayer=[500, 500, 2000],
                 decodeLayer=[2000, 500, 500])

    # Create trainer
    trainer = IDECTrainer(model, device)

    # Pretrain
    trainer.pretrain(dataloader, epochs=100)

    # Train IDEC
    trainer.train(dataloader, epochs=50, gamma=0.1)

    # Get results
    print("\n=== Getting Results ===")

    # 1. Hard cluster assignments
    y_pred = trainer.predict(dataloader)
    print(f"Hard assignments shape: {y_pred.shape}")

    # 2. Latent space representations
    latent_features = trainer.get_latent_representation(dataloader)
    print(f"Latent features shape: {latent_features.shape}")

    # 3. Soft cluster assignments (probabilities)
    soft_assignments = trainer.get_soft_assignments(dataloader)
    print(f"Soft assignments shape: {soft_assignments.shape}")

    # 4. Reconstructed data
    reconstructions = trainer.reconstruct(dataloader)
    print(f"Reconstructions shape: {reconstructions.shape}")

    # Evaluate clustering (if you have true labels for validation)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)

    print(f"\nClustering Results:")
    print(f"NMI: {nmi:.4f}")
    print(f"ARI: {ari:.4f}")

    # Example: Using latent features with traditional clustering
    print(f"\n=== Using Latent Features for Comparison ===")
    from sklearn.cluster import KMeans

    # Traditional K-means on latent features
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans_pred = kmeans.fit_predict(latent_features)

    kmeans_nmi = normalized_mutual_info_score(y_true, kmeans_pred)
    kmeans_ari = adjusted_rand_score(y_true, kmeans_pred)

    print(f"K-means on IDEC latent features:")
    print(f"NMI: {kmeans_nmi:.4f}")
    print(f"ARI: {kmeans_ari:.4f}")