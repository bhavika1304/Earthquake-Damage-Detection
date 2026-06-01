import torch
import numpy as np
import cv2

from model import BitemporalSiamese

# ---------------------------------------------------
# Device
# ---------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ---------------------------------------------------
# Model Paths
# ---------------------------------------------------

MODEL_PATHS = {
    "China": "models/china_best.pt",
    "Morocco": "models/morocco_best.pt",
    "Mexico": "models/mexico_best.pt"
}

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------

def load_model(region):

    model = BitemporalSiamese(
        num_classes=3,
        pretrained=False
    )

    checkpoint = torch.load(
        MODEL_PATHS[region],
        map_location=DEVICE,
        weights_only=False
    )

    if "model_state" in checkpoint:
        state = checkpoint["model_state"]
    elif "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint

    state = {k: v for k, v in state.items() if "cached_grid" not in k}

    model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    model.eval()

    return model

# ---------------------------------------------------
# Prepare Tensor
# ---------------------------------------------------

def prepare_tensor(img):
    img = cv2.resize(img, (512, 512))
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img  = (img - mean) / std
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    return tensor

# ---------------------------------------------------
# Region Default Preprocessing
# ---------------------------------------------------

def apply_region_default(pre_img, post_img, region):
    """
    Apply the preprocessing used during training for each region.
    This runs automatically before prediction regardless of what
    the user selects in the quality issue dropdown.

    China  — trained with Bright (brightness augmentation)
             no preprocessing needed at inference since the
             loader already handles normalisation.

    Morocco — trained with Blur preprocessing.
              Always apply gaussian blur before predicting.

    Mexico — trained with no special preprocessing.
    """
    """
    
    if region == "Morocco":
        from preprocessing.blur import apply_gaussian_blur
        pre_img  = apply_gaussian_blur(pre_img)
        post_img = apply_gaussian_blur(post_img)
    """
    return pre_img, post_img

# ---------------------------------------------------
# Predict Damage
# ---------------------------------------------------

def predict_damage(model, pre_img, post_img, region="China"):

    # apply training-time default preprocessing per region
    pre_img, post_img = apply_region_default(pre_img, post_img, region)

    pre_tensor  = prepare_tensor(pre_img)
    post_tensor = prepare_tensor(post_img)

    with torch.no_grad():

        output = model(pre_tensor, post_tensor)

        # model returns (logits, aux) — unpack if needed
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        pred = torch.argmax(logits, dim=1)

    return pred.squeeze().cpu().numpy()