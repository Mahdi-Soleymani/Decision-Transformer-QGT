import torch
from model_QGT import DecisionTransformer, GPTConfig

def test_attention_mask_no_leakage():
    B, k, T = 1, 2, 3  # batch size, UB dim, num time steps
    config = GPTConfig(rtg_dim=1, query_dim=k, query_result_dim=1, block_size=T, n_embd=16)
    model = DecisionTransformer(config)

    # Dummy inputs
    rtgs = torch.tensor([[[0.1], [0.2], [0.3]]])  # shape: [1, 3, 1]
    qrs = torch.tensor([[[0.5], [0.6], [0.7]]])    # shape: [1, 3, 1]
    queries = torch.rand(1, 3, k)
    mask_lengths = torch.tensor([3])  # all valid
    upper_bounds = torch.ones(1, k)

    # Forward hook to capture expanded mask
    expanded_masks = []

    def capture_attention_mask(self, x, attention_mask=None):
        if attention_mask is not None:
            expanded_masks.append(attention_mask.clone().detach())
        return self.resid_drop(self.proj(x))

    # Attach hook to first block's attention
    model.blocks[0].attn.forward = capture_attention_mask.__get__(model.blocks[0].attn)

    with torch.no_grad():
        _ = model(mask_lengths, rtgs.squeeze(-1), qrs.squeeze(-1), upper_bounds, queries)

    # Now expanded_masks[0] is shape [B, T, T], where T = k + 3 * block_size
    attn_mask = expanded_masks[0][0]  # remove batch dim

    # Compute the token indices:
    # [UB_1, ..., UB_k, RTG_1, QR_1, Q_1, RTG_2, QR_2, Q_2, RTG_3, QR_3, Q_3]
    # Say k = 2 => UB_0, UB_1 at [0,1], Q_1 is at idx 7, Q_2 at idx 10, Q_3 at idx 13

    q_indices = [k + 2 + i*3 for i in range(T)]  # 7, 10, 13

    print("Checking visible tokens for each Q_i:")
    for i, q_idx in enumerate(q_indices):
        visible = torch.nonzero(attn_mask[q_idx], as_tuple=False).squeeze(-1).tolist()
        print(f"Q_{i+1} at index {q_idx} sees tokens: {visible}")
        assert all(v <= q_idx for v in visible), f"Q_{i+1} sees future tokens!"

    print("✅ No leakage detected.")

test_attention_mask_no_leakage()
