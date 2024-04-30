# %%
from transformers import GPT2Tokenizer, GPT2Model
import torch

# %%
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

model = GPT2Model.from_pretrained('gpt2').to('mps')
model.eval()

tokenizer.pad_token = tokenizer.eos_token

def get_embedding_tensor(text):
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to('mps')
    with torch.no_grad():
        output = model(**encoded_input)
    token_embeddings = output.last_hidden_state
    token_embeddings = token_embeddings.mean(dim=1).squeeze()
    return token_embeddings

# %%
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # First layer goes from 768 -> 256
        # Second layer goes from 256 -> 64
        # Third layer goes from 64 -> 16
        # Fourth layer goes from 16 -> 1

        self.layer1 = nn.Linear(768, 256, bias=True)
        self.layer2 = nn.Linear(256, 64, bias=True)
        self.layer3 = nn.Linear(64, 16, bias=True)
        self.layer4 = nn.Linear(16, 1, bias=True)

        self.activation = nn.ReLU()

        self.layers = nn.Sequential(self.layer1, self.activation,
                                    self.layer2, self.activation,
                                    self.layer3, self.activation,
                                    self.layer4, nn.Sigmoid())        

    def forward(self, x):
        # x: (batch, 768)

        x = self.layers(x)
        return x

# %%
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.optim import AdamW

# %%
dataset = load_dataset("stanfordnlp/sst2")
dataset

# %%
dataset_train = dataset['train']
dataset_test = dataset['test']
dataset_validation = dataset['validation']

# %%
dataset_train[0]

# %%
train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset_validation, shuffle=False)
test_loader = DataLoader(dataset_test, shuffle=False)

# %%
mlp = Model()
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

mlp.to(device)

print(device)
print(mlp)

# %%
criterion = nn.BCELoss()
optimizer = AdamW(mlp.parameters(), lr=0.001)
epochs = 100

# %%
def train_loop(dataloader=train_loader, model=mlp, loss_fn=criterion, optimizer=optimizer):
    
    size = len(dataloader.dataset)

    model.train()
    for batch in iter(dataloader):

        # print("xxxx")
        # print(batch)
        # print("xxxx")

        # Data is idx, sentence, label
        text = batch['sentence']
        label = batch['label']

        # print(text)
        # print(label)

        X = get_embedding_tensor(text)
        X = X.to(device)
        y = label.to(device)

        pred = model(X)
        pred = pred.squeeze(1)

        print(y)
        print(pred)

        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss = loss.item()
        print(f"Loss: {loss}")

    return loss.item()


def validation_loop(dataloader=val_loader, model=mlp, loss_fn=criterion):
    
    model.eval()

    size = len(dataloader.dataset)

    val_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.to(device))
            val_loss += loss_fn(pred, y.to(device)).item()

    val_loss /= size
    return val_loss

# %%
# Training loop

early_stop_ctr = 0
curr_val_loss = 10000
prev_val_loss = 10000

train_losses = []
val_losses = []

for epoch in range(epochs):

    if early_stop_ctr > 5:
        break

    print(f"Epoch {epoch + 1}\n-------------------------------")
    train_loss = train_loop()
    val_loss = validation_loop()

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < prev_val_loss:
        early_stop_ctr = 0
    else:
        early_stop_ctr += 1

    prev_val_loss = curr_val_loss
    curr_val_loss = val_loss

# %%



