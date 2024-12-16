import os
import matplotlib.pyplot as plt
import numpy as np 
import torch 
import sys
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from models.cnn_attention import CNNWithAttention as CNN_model
import torch.optim as optim

# from models.cnnKan_attention import CNNWithAttention as CNN_model
# from models.dynamic_cnn_attention import CNN2D as CNN_model
# from models.dynamic_cnnKan_attention import CNN2D as CNN_model
# from models.multi_cnn_kan import CNNWithAttention as CNN_model

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

def train(config, train_loader, val_loader):

    activation_fn_map = {
        "ReLU": nn.ReLU,
        "LeakyReLU": nn.LeakyReLU,
        "ELU": nn.ELU
    }

    activation_fn = activation_fn_map.get(config["activation_fn"])
    if activation_fn is None:
        raise ValueError(f"Invalid activation function: {config['activation_fn']}")
    
    model = CNN_model(**config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.MSELoss()

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for images_org, images_crop in train_loader:
            images_org, images_crop = images_org.to(device), images_crop.to(device)
            optimizer.zero_grad()
            outputs = model(images_org)
            loss = criterion(outputs, images_crop)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images_org, images_crop in val_loader:
                images_org, images_crop = images_org.to(device), images_crop.to(device)
                outputs = model(images_org)
                loss = criterion(outputs, images_crop)
                val_loss += loss.item()
                
        # Update temperature if the method is available in the model
        if hasattr(model, 'update_temperature'):
            model.update_temperature()

        print(f"Epoch [{epoch+1}/10], Training Loss: {running_loss/len(train_loader):.4f}, Validation Loss: {val_loss/len(val_loader):.4f}")
        # Save the final model
    torch.save(model.state_dict(), 'final_model.pth')
    print("Model saved as 'final_model.pth'")

def load_model(config, model_path='final_model.pth'):
    model = CNN_model(**config).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, input_tensor):
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
    return output
