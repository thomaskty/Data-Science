import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(101)

# --------------------------
# Sample text preprocessing
# --------------------------
with open('data/fault_in_our_stars.txt') as file:
    text = file.read()

words = [i.replace('.', '') for i in text.replace('\n', ' ').split() if len(i) > 0]
words = [i.lower() for i in words]

unique_words = list(set(words))
index2word = {ind: word for ind, word in enumerate(unique_words)}
word2index = {j: i for i, j in index2word.items()}

# --------------------------
# n-gram training pairs
# --------------------------
def create_ngrams(words, n=3):
    """
    Create n-gram training pairs.
    Input: words (list of str), n (int, e.g., 3 for trigram)
    Output: list of (context, target) pairs
    """
    training_pairs = []
    for i in range(n-1, len(words)):
        context = words[i-(n-1):i]   # previous n-1 words
        target = words[i]            # next word
        context_tensor = torch.tensor([word2index[w] for w in context])
        target_tensor = torch.tensor(word2index[target])
        training_pairs.append((context_tensor, target_tensor))
    return training_pairs

N = 3  # Change this to any n for n-gram
training_pairs = create_ngrams(words, n=N)

# --------------------------
# Dataset & DataLoader
# --------------------------
class NGramDataset(Dataset):
    def __init__(self, training_pairs):
        self.training_pairs = training_pairs

    def __getitem__(self, idx):
        return self.training_pairs[idx]

    def __len__(self):
        return len(self.training_pairs)

trainset = NGramDataset(training_pairs)
train_loader = DataLoader(trainset, batch_size=16, shuffle=True)

# --------------------------
# Simple n-gram model
# --------------------------
class NGramLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # sum embeddings of previous words
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, x):
        # x: (batch, context_size)
        emb = self.embedding(x)  # (batch, context_size, embedding_dim)
        emb = emb.view(emb.size(0), -1)  # flatten (batch, context_size*embedding_dim)
        out = torch.relu(self.linear1(emb))
        out = self.linear2(out)
        return out

# --------------------------
# Training loop
# --------------------------
EMB_DIM = 128
VOCAB_SIZE = len(index2word)
model = NGramLM(vocab_size=VOCAB_SIZE, embedding_dim=EMB_DIM, context_size=N-1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

N_EPOCHS = 100

for epoch in range(N_EPOCHS):
    total_loss = 0
    for context, target in train_loader:
        optimizer.zero_grad()
        output = model(context)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")

# --------------------------
# Evaluation function
# --------------------------
@torch.no_grad()
def predict_next(context_words):
    context_idx = torch.tensor([word2index[w] for w in context_words]).unsqueeze(0)
    logits = model(context_idx)
    probs = torch.softmax(logits, dim=1)
    top_idx = torch.argmax(probs, dim=1)
    return index2word[top_idx.item()]

# Example
context_example = words[:N-1]
next_word = predict_next(context_example)
print("Context:", context_example)
print("Predicted next word:", next_word)
