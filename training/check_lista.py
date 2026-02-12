import torch
import numpy as np                                                                                                                                                                                                        
from train_lista import LISTANetwork
import data_generator                                                                                                                                                                                             
A, Y_val, X_val = data_generator.generate_dataset(m=250, n=500, k=25, num_samples=100, snr_db=40.0, seed=999)

model = LISTANetwork(500, 250, 16, A)
state = torch.load('lista_weights.pt', map_location='cpu', weights_only=True)
model.load_state_dict(state)
model.eval()

with torch.no_grad():
    y_t = torch.from_numpy(Y_val).float()
    x_t = torch.from_numpy(X_val).float()
    x_hat = model(y_t)
    mse = ((x_hat - x_t)**2).mean().item()
    sig = (x_t**2).mean().item()
    nmse_db = 10 * np.log10(mse / sig + 1e-15)
    print(f"Python LISTA NMSE = {nmse_db:.2f} dB")