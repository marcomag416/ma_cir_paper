import torch

def compute_clip_loss(image_features, text_features, logit_scale):
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logit_scale * text_features @ image_features.t()
    labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)

    image_loss = torch.nn.functional.cross_entropy(logits_per_image, labels)
    text_loss  = torch.nn.functional.cross_entropy(logits_per_text, labels)

    return (image_loss + text_loss) / 2

def compute_MA_bi_sw_loss(image_features, text_features, logit_scale, lambd=1, sim_grad=True, similarity="none"):
    loss1 = compute_MA_mono_sw_loss(image_features, text_features, logit_scale, lambd, sim_grad, similarity)
    loss2 = compute_MA_mono_sw_loss(text_features, image_features, logit_scale, lambd, sim_grad, similarity)
    return (loss1 + loss2) / 2


def compute_MA_mono_sw_loss(delta_features, target_features, logit_scale, lambd=1, sim_grad=True, similarity="none"):
    N = delta_features.shape[0]
    device = delta_features.device

    target_features = target_features / target_features.norm(dim=-1, keepdim=True)
    delta_features = delta_features / delta_features.norm(dim=-1, keepdim=True)

    # Expand for broadcasting: (N, N, D)
    source_i = target_features.unsqueeze(1)                # (N, 1, D)
    delta_j = delta_features.unsqueeze(0)                 # (1, N, D)
    delta_i = delta_features.unsqueeze(1)               # (N, 1, D)

    # Compute y for all i, j: y[i, j, :] = source_features[i] + lambd * (delta_features[j] - delta_features[i])
    y = source_i + lambd * (delta_j - delta_i)          # (N, N, D)
    y = y / y.norm(dim=-1, keepdim=True)         # Normalize along last dim

    # Reshape to (N*N, D)
    query = y.reshape(-1, y.shape[-1])

    # Labels: for each i, repeat torch.arange(N) N times
    labels = torch.arange(N, device=device).repeat(N)

    if similarity=="target":
        features_for_similarity = target_features
    elif similarity=="delta":
        features_for_similarity = delta_features
    elif similarity=="none":
        features_for_similarity = torch.ones((N*N,))
    else:
        raise ValueError(f"Invalid value for 'similarity'. Found {similarity}")

    #weights (w_ij)= text_features[i]*text_features[j] for each i,j
    features_for_similarity = features_for_similarity / features_for_similarity.norm(dim=-1, keepdim=True)  # Normalize features
    self_similarities = features_for_similarity @ features_for_similarity.T  # (N, N)
    weights = (self_similarities * (self_similarities > 0).int()).pow(2).flatten()  # Ensure non-negative weights, (N*N,)

    if not sim_grad:
        weights = weights.detach()  # Detach weights to prevent gradients from flowing through them

    # print("weights", weights.shape, weights.min(), weights.max())
    logits = logit_scale * query @ target_features.t()    # (N*N, N)
    unweighted_loss =  torch.nn.functional.cross_entropy(logits, labels, reduction='none')
    return (unweighted_loss * weights).sum() / weights.sum()

def build_loss_fn(loss_name, **kwargs):
    def loss_fn(outputs, labels, **kwargs):
        text_embeds = outputs["text_embeds"]
        vision_embeds = outputs["vision_embeds"]
        logit_scale = outputs["logit_scale"].to(text_embeds.device)


        with torch.autocast(text_embeds.device.type):
            if loss_name == "clip":
                return compute_clip_loss(vision_embeds, text_embeds, logit_scale)
            elif loss_name == "ma_bi_sw":
                return compute_MA_bi_sw_loss(vision_embeds, text_embeds, logit_scale, similarity="delta")
            elif loss_name == "ma_q2t_sw":
                return compute_MA_mono_sw_loss(vision_embeds, text_embeds, logit_scale, similarity="delta")
            elif loss_name == "ma_q2i_sw":
                return compute_MA_mono_sw_loss(text_embeds, vision_embeds, logit_scale, similarity="delta")
            elif loss_name == "ma_bi":
                return compute_MA_bi_sw_loss(vision_embeds, text_embeds, logit_scale, similarity="none")
            elif loss_name == "ma_q2t":
                return compute_MA_mono_sw_loss(vision_embeds, text_embeds, logit_scale, similarity="none")
            elif loss_name == "ma_q2i":
                return compute_MA_mono_sw_loss(text_embeds, vision_embeds, logit_scale, similarity="none")
            else:
                raise ValueError(f"Invalid value for 'loss_name'. Found {loss_name}")
        
    return loss_fn

    