import numpy as np
import torch
import torch.nn.functional as F


def generate_gradcam(model, input_tensor, target_class=None):
    """
    input_tensor: 1x1xHxW tensor (already normalized), requires_grad not needed by caller
    Returns: heatmap (HxW, values 0-1), predicted_class (int), probs (list)
    """
    model.eval()
    input_tensor = input_tensor.clone().requires_grad_(True)

    output = model(input_tensor)
    probs = F.softmax(output, dim=1)[0]
    if target_class is None:
        target_class = int(torch.argmax(probs).item())

    model.zero_grad()
    score = output[0, target_class]
    score.backward()

    activations = model.get_activations()[0]      # C x H x W
    gradients = model.get_gradients()[0]           # C x H x W

    weights = gradients.mean(dim=(1, 2))            # C
    cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * activations[i]

    cam = F.relu(cam)
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = cam.detach().numpy()

    # resize cam to input size
    H, W = input_tensor.shape[-2], input_tensor.shape[-1]
    cam_t = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
    cam_resized = F.interpolate(cam_t, size=(H, W), mode="bilinear", align_corners=False)
    heatmap = cam_resized.squeeze().numpy()

    return heatmap, target_class, probs.detach().numpy().tolist()
