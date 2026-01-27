from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import io
import traceback
import time

# Import custom modules
from model import SwinRGBHSV
from logic import generate_explanations, compute_agreement, run_quality_checks, generate_consensus_heatmap

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Running on device: {device}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES = [
    "Brown Blight", "Gray Blight", "Green mirid bug", "Healthy leaf", 
    "Helopeltis", "Red spider", "Tea algal leaf spot"
]

AGREEMENT_THRESHOLD_STRONG = 0.35

def get_model():
    print("Loading model weights...")
    model = SwinRGBHSV(
        model_name="swin_tiny_patch4_window7_224",
        num_classes=len(CLASSES),
        use_hsv_branch=False
    )
    try:
        checkpoint = torch.load("ema_model_weights_only.pth", map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict, strict=True)
        print("Model loaded (strict mode)")
    except Exception as e:
        print(f"Strict load failed: {e}. Trying non-strict...")
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded (non-strict mode)")
        
    model.to(device)
    model.eval()
    return model

model = get_model()

try:
    target_layer = model.backbone.layers[-1].blocks[-1].norm1
    print(f"Target layer: {target_layer}")
except:
    target_layer = model.backbone.norm
    print(f"Target layer (fallback): {target_layer}")
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    
    try:
        contents = await file.read()
        image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        original_img_np = np.array(image_pil) / 255.0

        input_tensor = transform(image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)
        
        predicted_label = CLASSES[predicted_idx.item()]
        conf_score = float(confidence.item())

        with torch.set_grad_enabled(True):
            explanations, masks = generate_explanations(
                model, input_tensor, target_layer, original_img_np
            )

        merged_image = generate_consensus_heatmap(masks, original_img_np)
        explanations["consensus"] = merged_image if merged_image else explanations.get("gradcam", "")

        agreement_score = compute_agreement(masks)
        quality_flags = run_quality_checks(masks, agreement_score)
        any_issue = any(quality_flags.values())
        is_low_agreement = agreement_score < AGREEMENT_THRESHOLD_STRONG
        final_status = "retake_required" if (is_low_agreement or any_issue) else "accepted"
        
        return {
            "prediction": {"label": predicted_label, "confidence": round(conf_score, 3)},
            "agreement_score": round(agreement_score, 3),
            "quality_flags": quality_flags,
            "explanations": explanations,
            "final_status": final_status
        }

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "classes": len(CLASSES),
        "device": str(device)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)