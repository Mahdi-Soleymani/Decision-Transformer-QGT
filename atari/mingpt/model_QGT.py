import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
logger = logging.getLogger(__name__)
import time

class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class GPTConfig:
    def __init__(self, rtg_dim, query_dim, query_result_dim, block_size, **kwargs):
        self.rtg_dim = rtg_dim
        self.query_dim = query_dim  # Dimensionality of the query
        self.query_result_dim = query_result_dim  # Dimensionality of the query result
        self.block_size = block_size  # Length of the input sequence
        self.upper_bound_dim =  query_dim # Add this
        self.n_layer = kwargs.get("n_layer", 6)
        self.n_head = kwargs.get("n_head", 8)
        self.n_embd = kwargs.get("n_embd", 128)
        self.embd_pdrop = kwargs.get("embd_pdrop", 0.1)
        self.resid_pdrop = kwargs.get("resid_pdrop", 0.1)
        self.attn_pdrop = kwargs.get("attn_pdrop", 0.1)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        #self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)))
        total_tokens = 4 * config.block_size + config.upper_bound_dim
        self.register_buffer("mask", torch.tril(torch.ones(total_tokens, total_tokens)).view(1, 1, total_tokens, total_tokens))

        # self.register_buffer("mask", torch.tril(torch.ones(3*config.block_size, 3*config.block_size))
        #                      .view(1, 1, 3*config.block_size, 3*config.block_size))  # Shape: (1, 1, block_size, block_size)

        # print(self.mask.shape)
        # time.sleep(60)

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        
        # Apply additional custom mask (if provided)
        # if attention_mask is not None:
        #     att = att.masked_fill(attention_mask[:, None, :, :] == 0, float('-inf'))  # Broadcast across heads


        if attention_mask is not None:
            # if attention_mask.dim() == 2:  # Padding mask [B, T]
            #     attention_mask = attention_mask.unsqueeze(1) & attention_mask.unsqueeze(2)  # [B, 1, T, T]
            attention_mask = attention_mask.bool()
            att = att.masked_fill(attention_mask[:, None, :, :] == 0, float('-inf'))





        att = torch.clamp(att, min=-1e4, max=1e4)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

class Block(nn.Module):

    def masked_layer_norm(self, x, layer_norm, attention_mask):
        # "" Applies LayerNorm while ignoring padded positions.
        # x: (B, T, C) tensor
        # attention_mask: (B, T) binary mask (1 for valid, 0 for padding) ""
        
        # Compute mean/std only on unmasked positions



        mean = (x * attention_mask).sum(dim=1, keepdim=True) / attention_mask.sum(dim=1, keepdim=True)
        std = torch.sqrt(((x - mean) ** 2 * attention_mask).sum(dim=1, keepdim=True) / attention_mask.sum(dim=1, keepdim=True) + 1e-5)

        # Normalize only valid positions
        x = (x - mean) / std
        x = layer_norm.weight * x + layer_norm.bias
        # print("x shape:", x.shape)
        # print("attention_mask shape before unsqueeze:", attention_mask.shape)
        # print("attention_mask shape after unsqueeze:", attention_mask.shape)
        # time.sleep(60)
        # Reapply mask to zero out padded positions
        return x * attention_mask


    
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd), GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd), nn.Dropout(config.resid_pdrop)
        )

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.mlp(self.ln2(x))

        # x = x + self.attn(self.masked_layer_norm(x, self.ln1, attention_mask))

        # x = x + self.mlp(self.ln2(x))
        return x

class DecisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Define embeddings for RTG, Query Result, and Query
        self.rtg_embedding = nn.Sequential(nn.Linear(config.rtg_dim, config.n_embd),nn.Tanh())
        self.query_result_embedding = nn.Sequential(nn.Linear(config.query_result_dim, config.n_embd),nn.Tanh())
        self.query_embedding = nn.Sequential(nn.Linear(config.query_dim, config.n_embd),nn.Tanh())
        self.upper_bound_embedding = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())
        self.entropy_embedding = nn.Sequential(nn.Linear(1, config.n_embd),nn.Tanh())
        nn.init.normal_(self.query_embedding[0].weight, mean=0.0, std=0.02)
        nn.init.normal_(self.entropy_embedding[0].weight, mean=0.0, std=0.02)

        # self.timestep_embedding = nn.Embedding(config.block_size, config.n_embd)


        # Positional embedding
        #self.pos_emb = nn.Parameter(torch.zeros(1, int(3*config.block_size), config.n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, 4*config.block_size + config.upper_bound_dim, config.n_embd))

        # Dropout
        self.drop = nn.Dropout(config.embd_pdrop)
        self.token_ln = nn.LayerNorm(config.n_embd)


        # Transformer blocks
        #self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # Final LayerNorm and output head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.query_dim,bias=False)  # Action dim to predict next action
        self.block_size = config.block_size
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))


    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if  module.bias is not None:
                module.bias.data.zero_()

        if isinstance(module,  nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1)
 
        
## original 

        # if isinstance(module, (nn.Linear, nn.Embedding)):
        #     module.weight.data.normal_(mean=0.0, std=0.02)
        #     if isinstance(module, nn.Linear) and module.bias is not None:
        #         module.bias.data.zero_()        

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, config):
    # Define the optimizer (e.g., Adam)
        optimizer = optim.AdamW(self.parameters(), lr=config.learning_rate)
        optimizer = torch.optim.AdamW(
        self.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
)

        # optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        #optimizer = torch.optim.SGD(self.parameters(), lr=config.learning_rate)

        return optimizer  # Or return (optimizer, scheduler) if you use a scheduler



    def forward(self, mask_length, rtgs,  query_results, upper_bounds, entropy, queries=None,targets=None):
        # Encode the inputs (rtg, query, query result)
        # print(f"rtgs is {rtgs.shape}")
        # time.sleep(60)
 
        # print(masks[1:3])
        # print(triplet_mask[1:3])
        # time.sleep(30)

        # Assuming triplet_mask should be of shape (batch_size, sequence_length)
        # Initialize the mask tensor with zeros first
      
        
        rtgs = rtgs.unsqueeze(-1)  
        rtg_embeddings = self.rtg_embedding(rtgs.float()) 
        query_results=query_results.unsqueeze(-1) 
        query_result_embeddings = self.query_result_embedding(query_results.float())
        entropy = entropy.unsqueeze(-1)  # shape: [B, T, 1]
        entropy_embeddings = self.entropy_embedding(entropy.float())
        quad_mask = torch.zeros((query_results.shape[0], query_results.shape[1]*4 , 1), dtype=torch.int, device=query_result_embeddings.device)
        
        if queries is not None : 
            query_embeddings = self.query_embedding(queries)
            token_embeddings = torch.zeros((query_results.shape[0], query_results.shape[1]*4 , self.config.n_embd), dtype=torch.float32, device=query_result_embeddings.device)
            token_embeddings[:, ::4, :] = rtg_embeddings
            token_embeddings[:, 1::4, :] = entropy_embeddings
            token_embeddings[:, 2::4, :] = query_result_embeddings
            token_embeddings[:, 3::4, :] = query_embeddings

            for i, length in enumerate(mask_length):
                quad_mask[i, :int((4 * length ).item())-1] = 1  # Set valid part to 1
                quad_mask[i, int((4 * length ).item())-1:] = 0  # Set the remaining part to 0
           
            #print(triplet_mask)
            # time.sleep(60)
        
        elif queries is None:  # only happens at very first timestep of evaluation
            rtg_embeddings = self.rtg_embedding(rtgs.type(torch.float32))
            token_embeddings = torch.zeros((query_results.shape[0], query_results.shape[1]*3, self.config.n_embd), dtype=torch.float32, device=query_result_embeddings.device)
            token_embeddings[:,::2,:] = rtg_embeddings # really just [:,0,:]
            token_embeddings[:,1::2,:] = query_result_embeddings # really just [:,1,:]
            quad_mask=torch.zeros(4*query_results.shape[1])
            quad_mask[2*mask_length]=1



        # Ensure upper_bounds is always 2D before unsqueeze
        if upper_bounds.ndim == 3 and upper_bounds.shape[-1] == 1:
            upper_bounds = upper_bounds.squeeze(-1)
        elif upper_bounds.ndim != 2:
            raise ValueError(f"upper_bounds should be 2D [B, k], got {upper_bounds.shape}")




        upper_bounds = upper_bounds.unsqueeze(-1) # shape: (B, k, D)
        upper_bounds_embedding = self.upper_bound_embedding(upper_bounds)    
        token_embeddings = torch.cat([upper_bounds_embedding, token_embeddings], dim=1)
        ub_mask = torch.ones((quad_mask.shape[0], upper_bounds.shape[1]), dtype=quad_mask.dtype, device=quad_mask.device)
        ub_mask = ub_mask.unsqueeze(-1)
        quad_mask = torch.cat([ub_mask, quad_mask], dim=1)  # shape becomes (B, T+ub_len)
        #expanded_mask = triplet_mask.unsqueeze(1) & triplet_mask.unsqueeze(2)


        # Add positional embeddings


        position_embeddings = self.pos_emb[:, :token_embeddings.shape[1], :]

        #x = self.drop(token_embeddings + position_embeddings)
        x = self.token_ln(token_embeddings + position_embeddings)
        x = self.drop(x)

     
        if quad_mask.dim() == 3:  # [B, T, 1]
            quad_mask = quad_mask.squeeze(-1)  # [B, T]

        # Now expand to [B, T, T]

        # triplet_mask is shape [B, 3 * seq_len]
        # upper_bounds_mask = ones of shape [B, k]

        expanded_mask = quad_mask.unsqueeze(1) & quad_mask.unsqueeze(2)
        #expanded_mask = expanded_mask.bool()

        # print(expanded_mask[0]) 
        # print(expanded_mask.shape) 
        # time.sleep(100)

        # Pass through transformer blocks
        #x = self.blocks(x,triplet_mask)
        for block in self.blocks:
            x = block(x, expanded_mask)
            #x = block(x)

        # print(x)
        # time.sleep(100)

        # # Final LayerNorm
        # x = self.ln_f(x)

        # Output layer to predict next action (query)
        logits = self.head(x)

        #time.sleep(1)
        if queries is not None:
            #logits = logits[:, 1::3, :] # only keep predictions from query_embeddings
            logits = logits[:, upper_bounds.shape[1] + 2::4, :]  # Adjusted for upper_bounds prefix

        # elif queries is None:
        #     logits = logits[:, 1:, :]
            
        
        probabilities = torch.sigmoid(logits)  # Apply sigmoid to get probabilities in range [0, 1]
        # print(targets.shape)
        # print(logits.shape)
        # time.sleep(100)
        loss = None
        if targets is not None:
            if self.config.label_smoothing > 0.0:
                targets = targets * (1 - self.config.label_smoothing) + 0.5 * self.config.label_smoothing


            if self.config.criterion=="bce":
                loss_fn = nn.BCEWithLogitsLoss(reduction="none")
                loss = loss_fn(logits, targets)
            if self.config.criterion=="mse":
                loss= nn.MSELoss(reduction="none")(probabilities, targets)
            if self.config.criterion=="mae":
                loss_fn = torch.nn.L1Loss(reduction="none")          # MAE
                loss = loss_fn(probabilities, targets)
            if self.config.criterion=="hub":
                loss_fn = torch.nn.L1Loss(reduction="none")          # MAE
                loss = loss_fn(probabilities, targets)

            
             

            # Create mask of shape (block_size, k)
            mask = torch.zeros_like(logits, device=logits.device)
            for i in range(logits.shape[0]):
                mask[i,:mask_length[i]-1, :] = 1  # Set first mask_length rows to 1

            # Apply mask: Keep only the relevant losses

            loss = loss * mask  # Zero out the masked parts
            # print(loss)
            # l=2
            # print(mask_length[l])
            # print(targets[l])
            # print(query_results[l])
            # print(rtgs[l])
            # time.sleep(100)
            # Normalize the loss by the number of valid elements
        


            if mask.sum() == 0:
                loss = torch.tensor(0.0, device=logits.device)
            else:
                loss = loss.sum() / mask.sum()
            #loss = loss.sum() / mask.sum()
            #loss = loss.sum() 
            


        return probabilities,loss
