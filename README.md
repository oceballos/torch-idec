# Torch IDEC
implementartion of Improved Deep Embedded Clustering with Local Structure Preservation from of Xifeng Guo et.al in Pytorch, original project implemented in Keras by its original authors.



## Usage

!pip install torch

~~~python
from idectorch import IDEC, IDECTrainer 

# Create trainer
trainer = IDECTrainer(model, device)

# Pretrain
trainer.pretrain(dataloader, epochs=100)

# Train IDEC
trainer.train(dataloader, epochs=50, gamma=0.1)

# Get predictions
y_pred = trainer.predict(dataloader)

~~~

## Dependencies
Scikit Learn
PyTorch

