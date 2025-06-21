import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self, z_dim=64, label_dim=10, img_dim=784):
        super().__init__()
        self.label_emb = nn.Embedding(10, label_dim)
        self.model = nn.Sequential(
            nn.Linear(z_dim + label_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, img_dim),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_input = self.label_emb(labels)
        x = torch.cat([z, label_input], dim=1)
        img = self.model(x)
        return img


# Load model
device = torch.device("cpu")
model = Generator()
model.load_state_dict(torch.load(
    "/Users/mac/Desktop/challenge/digit/generator_improved.pth", map_location=device))

model.eval()

st.title("Digit Generator (0–9)")

digit = st.number_input("Choose a digit to generate (0-9):",
                        min_value=0, max_value=9, step=1)

if st.button("Generate Images"):
    z_dim = 64
    with torch.no_grad():
        z = torch.randn(5, z_dim)
        labels = torch.full((5,), digit, dtype=torch.long)
        images = model(z, labels).view(-1, 28, 28)

    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axs[i].imshow(images[i], cmap='gray')
        axs[i].axis('off')
    st.pyplot(fig)
