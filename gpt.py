import random
import torch

random.seed(42)
torch.manual_seed(42)

docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars) # token id for a special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")

# Initialize the parameters, to store the knowledge of the model
n_layer = 1     # depth of the transformer neural network (number of layers)
n_embd = 16     # width of the network (embedding dimension)
block_size = 16 # maximum context length of the attention window (note: the longest name is 15 characters)
n_head = 4      # number of attention heads
head_dim = n_embd // n_head # derived dimension of each head
matrix = lambda nout, nin, std=0.08: torch.nn.Parameter(torch.randn(nout, nin) * std)
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = list(state_dict.values())
print(f"num params: {sum(p.numel() for p in params)}")

# Define the model architecture: a function mapping tokens and parameters to logits over what comes next
# Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
def rmsnorm(x):
    scale = ((x * x).mean() + 1e-5) ** -0.5
    return x * scale

def gpt(token_id, pos_id, keys, values):
    x = state_dict['wte'][token_id] + state_dict['wpe'][pos_id] # token + position embedding
    x = rmsnorm(x) # note: not redundant due to backward pass via the residual connection

    for li in range(n_layer):
        # 1) Multi-head Attention block
        x_residual = x
        x = rmsnorm(x)
        q = state_dict[f'layer{li}.attn_wq'] @ x
        k = state_dict[f'layer{li}.attn_wk'] @ x
        v = state_dict[f'layer{li}.attn_wv'] @ x
        keys[li].append(k)
        values[li].append(v)
        x_attn_parts = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = torch.stack([ki[hs:hs+head_dim] for ki in keys[li]])  # (T, head_dim)
            v_h = torch.stack([vi[hs:hs+head_dim] for vi in values[li]])  # (T, head_dim)
            attn_weights = torch.softmax(k_h @ q_h / head_dim**0.5, dim=0)  # (T,)
            x_attn_parts.append(v_h.T @ attn_weights)                        # (head_dim,)
        x = state_dict[f'layer{li}.attn_wo'] @ torch.cat(x_attn_parts)
        x = x + x_residual
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = torch.relu(state_dict[f'layer{li}.mlp_fc1'] @ x)
        x = state_dict[f'layer{li}.mlp_fc2'] @ x
        x = x + x_residual

    return state_dict['lm_head'] @ x

# Let there be Adam, the blessed optimizer
learning_rate, beta1, beta2 = 0.01, 0.85, 0.99
optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(beta1, beta2), eps=1e-8)

# Repeat in sequence
num_steps = 1000 # number of training steps
for step in range(num_steps):

    # Take single document, tokenize it, surround it with BOS special token on both sides
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # Forward the token sequence through the model, building up the computation graph all the way to the loss
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = torch.softmax(logits, dim=0)
        losses.append(-probs[target_id].log())
    loss = sum(losses) / n # final average loss over the document sequence. May yours be low.

    # Backward the loss, calculating the gradients with respect to all model parameters
    optimizer.zero_grad()
    loss.backward()

    # Adam optimizer update with linear learning rate decay
    for g in optimizer.param_groups:
        g['lr'] = learning_rate * (1 - step / num_steps)
    optimizer.step()

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.item():.4f}", end='\r')

# Inference: may the model babble back to us
temperature = 0.5 # in (0, 1], control the "creativity" of generated text, low to high
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    with torch.no_grad():
        for pos_id in range(block_size):
            logits = gpt(token_id, pos_id, keys, values)
            probs = torch.softmax(logits / temperature, dim=0)
            token_id = random.choices(range(vocab_size), weights=probs.tolist())[0]
            if token_id == BOS:
                break
            sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")