import torch
from torch.distributions import Categorical
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
import wandb

NUM_EPOCHS = 100
BATCH_SIZE = 1000
NUM_TOKENS = 10
LR = 1e-5
KL_FACTOR = 6000
WANDB = False

if WANDB:
    run = wandb.init(
        project="tinycatstories",
        config={
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "num_tokens": NUM_TOKENS,
            "learning_rate": LR,
            "kl_factor": KL_FACTOR,
        },
    )

embedding_model = SentenceTransformer("all-MiniLM-L6-v2").to("cuda")
reference_embedding = embedding_model.encode("cat", convert_to_tensor=True)

for param in embedding_model.parameters():
    param.requires_grad = False


def compute_rewards(sequences):
    sequence_embeddings = embedding_model.encode(sequences, convert_to_tensor=True)
    cosine_similarities = util.pytorch_cos_sim(
        reference_embedding.unsqueeze(0), sequence_embeddings
    ).squeeze()
    return cosine_similarities


model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M").to("cuda")
ref_model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M").to(
    "cuda"
)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
optimizer = AdamW(model.parameters(), lr=LR)

for param in ref_model.parameters():
    param.requires_grad = False

prompt = "Once upon a time there was"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

for epoch in range(NUM_EPOCHS):
    model.train()

    output_ids = torch.full(
        (BATCH_SIZE, NUM_TOKENS), tokenizer.eos_token_id, device="cuda"
    )
    output_ids[:, : input_ids.shape[1]] = input_ids

    log_probs_accumulated = torch.zeros((BATCH_SIZE, 1), device="cuda")
    kl_div_accumulated = torch.zeros((BATCH_SIZE, 1), device="cuda")

    # keep track of which stories (within the batch) have completed
    # when a story is complete there is an EOS token
    # we must stop accumulating log_probs and kl divergence for that story
    # this only happens > 200 tokens
    active_mask = torch.ones(BATCH_SIZE, dtype=torch.bool, device="cuda")

    for i in range(input_ids.shape[1], NUM_TOKENS):
        prompt = output_ids[:, :i].clone()
        logits = model(prompt).logits[:, -1, :]
        # Only consider logits of active sequences
        logits_active = logits[active_mask]
        if logits_active.shape[0] == 0:
            # All sequences are finished
            break
        probs = torch.nn.functional.softmax(logits_active, dim=-1)
        dist = Categorical(probs)
        next_tokens = dist.sample()
        log_probs_accumulated[active_mask] += dist.log_prob(next_tokens).unsqueeze(-1)
        output_ids[active_mask, i] = next_tokens

        # Compute reference model
        ref_logits = ref_model(prompt).logits[:, -1, :]
        ref_logits_active = ref_logits[active_mask]

        # Compute KL Divergence
        kl_div = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(logits_active, dim=-1),
            torch.nn.functional.log_softmax(ref_logits_active, dim=-1),
            reduction="none",
            log_target=True,
        )
        kl_div_accumulated[active_mask] += kl_div.mean(dim=-1).unsqueeze(-1)

        finished = next_tokens == tokenizer.eos_token_id
        active_indices = torch.nonzero(active_mask).squeeze(-1)
        new_mask = active_mask.clone()
        new_mask[active_indices] = ~finished
        active_mask = new_mask

    normalized_log_probs = log_probs_accumulated / NUM_TOKENS
    normalized_kl_div = kl_div_accumulated / NUM_TOKENS

    # Compute rewards for the entire batch
    with torch.no_grad():
        sequences = [
            tokenizer.decode(input_id, skip_special_tokens=True)
            for input_id in output_ids
        ]
        rewards = compute_rewards(sequences)

    # Compute loss for the entire batch
    neg_advantage = (-normalized_log_probs * rewards.unsqueeze(-1)).mean()
    loss = neg_advantage + KL_FACTOR * normalized_kl_div.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if WANDB:
        wandb.log(
            {"loss": loss, "reward": rewards.mean(), "kl": normalized_kl_div.mean()}
        )

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}: Loss: {loss.item()} Rewards: {rewards.mean()} NegAdv: {neg_advantage} KL: {normalized_kl_div.mean()}"
    )


save_directory = "./checkpoints"
model.save_pretrained(save_directory)
