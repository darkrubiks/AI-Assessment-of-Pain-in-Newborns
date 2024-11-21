import torch

def rollout(attentions, discard_ratio):
    result = torch.eye(attentions[0].size(-1))
    
    with torch.no_grad():
        for attention in attentions:
            #attention = attention.unsqueeze(dim=1)
            # don't drop the class token
            flat = attention.view(attention.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention.size(-1))
            a = (attention + 1.0*I)/2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)

    

    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0, 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    return mask   

class VITAttentionRollout:
    def __init__(self, model, discard_ratio=0.9):
        self.model = model
        self.discard_ratio = discard_ratio
        for blk in model.ViT.encoder.layers:
            blk.self_attention.register_forward_hook(self.get_attention)

        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output[1].cpu())

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)

        return rollout(self.attentions, self.discard_ratio)