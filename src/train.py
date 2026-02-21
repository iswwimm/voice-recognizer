import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import VoiceTripletDataset
from model import SpeakerEncoder

EPOCHS = 30         
BATCH_SIZE = 8      
LEARNING_RATE = 0.001
SAVE_PATH = "voice_auth_model.pth"

MY_VOICE_DIR = "data/my_voices"
OTHER_VOICE_DIR = "data/other_voices"

def train():
    print("Loading data...")
    dataset = VoiceTripletDataset(MY_VOICE_DIR, OTHER_VOICE_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SpeakerEncoder().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    print("Starting training...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        
        for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            optimizer.zero_grad()
            
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)
            
            loss = criterion(anchor_out, positive_out, negative_out)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")
        
        torch.save(model.state_dict(), SAVE_PATH)

    print(f"Training complete. Model saved to '{SAVE_PATH}'")

if __name__ == "__main__":
    train()