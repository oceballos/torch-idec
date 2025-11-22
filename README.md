# Torch IDEC
implementartion of Improved Deep Embedded Clustering with Local Structure Preservation from of Xifeng Guo et.al in Pytorch, original project implemented in Keras by its original authors.



## Usage
install pytorch if you didn't beforehand.
~~~batch
!pip install torch
~~~
Clone repository
~~~terminaloutput
!git clone https://github.com/oceballos/torch-idec.git
~~~

Add directory to PATH

~~~python
import sys
import os

py_file_location = "./torch-idec/"
sys.path.append(os.path.abspath(py_file_location))
~~~

Usage example, consider df to be a pandas dataframe.
~~~python
import torch
from torch.utils.data import DataLoader, TensorDataset
from idectorch import IDEC, IDECTrainer 

#check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#Convert your data to tensor
X_tensor = torch.FloatTensor(df.to_numpy())
dataset = TensorDataset(X_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

#Initialize model
model = IDEC(input_dim=df.shape[1], z_dim=16, n_clusters=8,
            encodeLayer=[50, 50, 200],
            decodeLayer=[200, 50, 50])

# Create trainer
trainer = IDECTrainer(model, device)

# Pretrain
trainer.pretrain(dataloader, epochs=100)

# Train IDEC
trainer.train(dataloader, epochs=50, gamma=0.1)

# Get predictions (hard assigned clusters)
y_pred = trainer.predict(dataloader)

# Get latent space
latent_features = trainer.get_latent_representation(dataloader)

# Get Soft cluster assignments (probabilities)
soft_assignments = trainer.get_soft_assignments(dataloader)

#Get Reconstructed data
reconstructions = trainer.reconstruct(dataloader)

~~~

## Dependencies
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.2.0
scipy>=1.10.0

## References
Guo, Xifeng et al. “Improved Deep Embedded Clustering with Local Structure Preservation.” International Joint Conference on Artificial Intelligence (2017).

## Licence
MIT

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
