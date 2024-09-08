from unet import SimpleUNet, UNet
from dataset import MUSDBDataset, audio_to_spectrogram, spectrogram_to_audio
import musdb
import torch
import torchaudio
import librosa
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Early Stopping Parameters
patience = 4
best_val_loss = float('inf')
epochs_without_improvement = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 256
num_epochs = 1
lr = 1e-3

# Function to perform validation
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for mix, vocals in val_loader:
            mix, vocals = mix.to(device), vocals.to(device)
            output = model(mix)
            
            # Match tensor sizes
            output, vocals = match_tensor_sizes(output, vocals)
            
            loss = criterion(output, vocals)
            val_loss += loss.item()
    return val_loss / len(val_loader)


# Updated training function with validation and early stopping
def train(
    model, 
    train_loader, 
    val_loader, 
    num_epochs=10, 
    lr=0.001, 
    patience=10
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    global best_val_loss, epochs_without_improvement
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        epoch_loss = 0.0
        
        for mix, vocals in train_loader:
            mix, vocals = mix.to(device), vocals.to(device)

            optimizer.zero_grad()
            output = model(mix)  # Forward pass
            
            
            # Match tensor sizes
            output, vocals = match_tensor_sizes(output, vocals)
            
            loss = criterion(output, vocals)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            
            epoch_loss += loss.item()
        
        
        # Check if validation loss improved
        if epoch % 5 == 0:
            model.eval()
            # Perform validation
            val_loss = validate(model, val_loader, criterion)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader)}, Val Loss: {val_loss}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                print(f"Validation loss improved to {best_val_loss}. Saving model.")
                torch.save(model.state_dict(), 'best_model.pth')  # Save best model
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement} epochs.")
            
            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break





def match_tensor_sizes(input_tensor, target_tensor):
    """ Match the sizes of input and target tensors by either truncating or padding """
    min_length = min(input_tensor.shape[-1], target_tensor.shape[-1])  # Find the minimum length
    
    # Truncate both tensors to the minimum length along the time dimension
    input_tensor = input_tensor[..., :min_length]
    target_tensor = target_tensor[..., :min_length]
    
    return input_tensor, target_tensor



# Test the model on a new audio track
def test_model(model, track):
    model.eval()
    
    audio_mix = track.audio.sum(axis=1)  # Mix audio (mono)
    spec_mix = audio_to_spectrogram(audio_mix)  # Convert to spectrogram
    mix_tensor = torch.tensor(spec_mix).unsqueeze(0).unsqueeze(0).float().cuda()  # To tensor
    
    # Forward pass to get the separated vocals
    with torch.no_grad():
        pred_spec = model(mix_tensor).squeeze().cpu().numpy()  # Remove batch and channel dim
    
    # Convert predicted spectrogram back to audio
    pred_audio = spectrogram_to_audio(pred_spec)
    
    # Plot original and separated spectrogram
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Mix')
    plt.imshow(np.log1p(spec_mix), aspect='auto', origin='lower')
    
    plt.subplot(1, 2, 2)
    plt.title('Predicted Vocals')
    plt.imshow(np.log1p(pred_spec), aspect='auto', origin='lower')
    plt.show()
    
    return pred_audio





# Splitting dataset into training and validation sets
mus = musdb.DB(subsets='train', split='train', download=True)
dataset = MUSDBDataset(mus)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
model = UNet().to(device) #SimpleUNet().to(device)

# Train with early stopping
train(model, train_loader, val_loader, num_epochs=num_epochs, lr=lr, patience=patience)




# Run the model on a test track
test_track = mus.load_mus_tracks(subsets="test")[0]
pred_vocals_audio = test_model(model, test_track)
