# Example usage

from idectorch import IDEC, IDECTrainer
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import torch
from torch.utils.data import DataLoader, TensorDataset

if __name__ == "__main__":
    # Generate sample data (you would replace this with your actual data)
    from sklearn.datasets import make_blobs

    # Create sample dataset
    X, y_true = make_blobs(n_samples=1000, centers=4, n_features=20,
                           random_state=42, cluster_std=1.5)

    # Convert to PyTorch
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y_true)
    dataset = TensorDataset(X_tensor, y_tensor)
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

    # Get predictions
    y_pred = trainer.predict(dataloader)

    # Evaluate
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)

    print(f"\nResults:")
    print(f"NMI: {nmi:.4f}")
    print(f"ARI: {ari:.4f}")