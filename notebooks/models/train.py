from torch.optim import Adam
from models.vae import loss_function
from tqdm import tqdm

def train_vae(model, dataloader, epochs=10, learning_rate=1e-3, beta=1.0):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    overall_loss = []
    for epoch in tqdm(range(epochs)):
        train_loss = 0
        for batch_data in dataloader:
            # If your dataset provides data in the form (data, labels), use batch_data[0]
            # If not, just use batch_data
            data = batch_data[0].float()
            
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar, beta)
            
            loss.backward()
            train_loss += loss.item()
            
            optimizer.step()
        overall_loss.append(train_loss/len(dataloader.dataset))
    return overall_loss