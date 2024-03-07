import kornia.losses
import torch
from torch.utils.data import DataLoader
from loader.tno import TNO
from model import FusionNet
from torch.optim import Adam
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ir_dir = 'data/tno/ir'
vi_dir = 'data/tno/vi'
dataset = TNO(ir_dir, vi_dir)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model = FusionNet().to(device)

criterion = kornia.losses.SSIMLoss(window_size=11)
optimizer = Adam(model.parameters(), lr=0.001)

losses = []

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (ir_images, vi_images) in enumerate(dataloader):
        ir_images, vi_images = ir_images.to(device), vi_images.to(device)

        optimizer.zero_grad()

        outputs = model(ir_images, vi_images)

        ir_loss = 1 - criterion(outputs, ir_images)
        vi_loss = 1 - criterion(outputs, vi_images)
        total_loss = ir_loss + vi_loss

        total_loss.backward()

        optimizer.step()

        running_loss += total_loss.item()

    average_loss = running_loss / len(dataloader)
    losses.append(average_loss)
    print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}')

plt.figure(figsize=(10, 5))
plt.plot(losses, label='Average Loss')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()

model_path = 'fusion_model.pth'
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')
