import cv2
import numpy as np
import torch
import base64
import itertools
from scipy import ndimage

# Import ALL methods
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit

def tensor_to_cv2(tensor):
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    return img

def encode_image(cv2_img):
    _, buffer = cv2.imencode(".png", cv2_img * 255)
    return base64.b64encode(buffer).decode("utf-8")

def normalize_heatmap(heatmap):
    h_min = heatmap.min()
    h_max = heatmap.max()
    if h_max - h_min < 1e-8:
        return np.zeros_like(heatmap)
    return (heatmap - h_min) / (h_max - h_min)

def swin_reshape_transform(tensor):
    if len(tensor.shape) == 4:
        return tensor.permute(0, 3, 1, 2)
    height = width = int(np.sqrt(tensor.shape[1]))
    result = tensor.transpose(1, 2)
    result = result.reshape(tensor.shape[0], tensor.shape[2], height, width)
    return result

class HiResCAM:
    def __init__(self, model, target_layers, reshape_transform=None):
        self.model = model
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.activations = None
        self.gradients = None
        
    def __call__(self, input_tensor):
        self.model.eval()
        self.activations = None
        self.gradients = None
        
        def forward_hook(module, input, output):
            if self.reshape_transform:
                self.activations = self.reshape_transform(output)
            else:
                self.activations = output
                
        def backward_hook(module, grad_input, grad_output):
            if self.reshape_transform:
                self.gradients = self.reshape_transform(grad_output[0])
            else:
                self.gradients = grad_output[0]
        
        handle_forward = self.target_layers[0].register_forward_hook(forward_hook)
        handle_backward = self.target_layers[0].register_full_backward_hook(backward_hook)
        
        try:
            output = self.model(input_tensor)
            target_class = output.argmax(dim=1)
            self.model.zero_grad()
            loss = output[0, target_class]
            loss.backward(retain_graph=True)
            
            activations = self.activations.detach().cpu().numpy()
            gradients = self.gradients.detach().cpu().numpy()
            
            element_wise = activations * gradients
            cam = element_wise.sum(axis=1)
            cam = np.maximum(cam, 0)
            
            if len(cam.shape) == 3:
                cam = cam[0]
            
            return cam
            
        finally:
            handle_forward.remove()
            handle_backward.remove()
    
    def release(self):
        pass

def generate_explanations(model, input_tensor, target_layer, original_img):
    print("Starting XAI generation...")
    results = {}
    masks = {}

    cams = {
        "gradcam": GradCAM(model=model, target_layers=[target_layer], reshape_transform=swin_reshape_transform),
        "gradcampp": GradCAMPlusPlus(model=model, target_layers=[target_layer], reshape_transform=swin_reshape_transform),
        "layercam": LayerCAM(model=model, target_layers=[target_layer], reshape_transform=swin_reshape_transform),
        "shap": AblationCAM(
            model=model, 
            target_layers=[target_layer], 
            reshape_transform=swin_reshape_transform, 
            ablation_layer=AblationLayerVit()
        ),
        "hirescam": HiResCAM(model=model, target_layers=[target_layer], reshape_transform=swin_reshape_transform)
    }

    orig_h, orig_w, _ = original_img.shape

    try:
        for name, cam_extractor in cams.items():
            try:
                print(f"Generating {name.upper()}...")
                if name == "shap":
                    cam_extractor.batch_size = 2
                
                if name == "hirescam":
                    grayscale_cam = cam_extractor(input_tensor=input_tensor)
                else:
                    grayscale_cam = cam_extractor(input_tensor=input_tensor)[0, :]
                
                normalized_heatmap = normalize_heatmap(grayscale_cam)
                resized_heatmap = cv2.resize(normalized_heatmap, (orig_w, orig_h))
                masks[name] = resized_heatmap

                visualization = show_cam_on_image(original_img, resized_heatmap, use_rgb=True)
                results[name] = encode_image(visualization)
                
            except Exception as e:
                print(f"Error in {name}: {e}")
                results[name] = ""
                masks[name] = np.zeros((orig_h, orig_w))
    finally:
        for cam_extractor in cams.values():
            if hasattr(cam_extractor, "activations_and_grads"):
                cam_extractor.activations_and_grads.release()
            else:
                cam_extractor.release()

    print("XAI generation complete.")
    return results, masks

def generate_consensus_heatmap(masks, original_img):
    print("Generating consensus heatmap...")
    valid_masks = [m for k, m in masks.items() if m.max() > 0]
    if not valid_masks: 
        return ""
    
    all_masks = np.array(valid_masks)
    consensus_mask = np.mean(all_masks, axis=0)
    consensus_mask = normalize_heatmap(consensus_mask)
    
    visualization = show_cam_on_image(original_img, consensus_mask, use_rgb=True)
    return encode_image(visualization)

def compute_agreement(masks, k=0.10):
    valid_masks = {k: m for k, m in masks.items() if m.max() > 0}
    if not valid_masks: return 0.0
    
    binary_masks = []
    for key, mask in valid_masks.items():
        threshold = np.percentile(mask, 100 - (k * 100))
        binary = (mask >= threshold).astype(float)
        binary_masks.append(binary)

    ious = []
    for m1, m2 in itertools.combinations(binary_masks, 2):
        intersection = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()
        iou = intersection / union if union > 0 else 0.0
        ious.append(iou)
    
    return float(np.mean(ious)) if ious else 0.0

def run_quality_checks(masks, agreement_score, k=0.10):
    valid_masks = {k: m for k, m in masks.items() if m.max() > 0}
    if not valid_masks: return {"background": False, "spread": False, "border": False}
        
    h, w = list(valid_masks.values())[0].shape
    h_start, h_end = int(h * 0.2), int(h * 0.8)
    w_start, w_end = int(w * 0.2), int(w * 0.8)
    
    center_fails = 0
    for key, mask in valid_masks.items():
        threshold = np.percentile(mask, 100 - (k * 100))
        binary_mask = (mask >= threshold).astype(float)
        total = binary_mask.sum()
        if total == 0: continue
        center_energy = binary_mask[h_start:h_end, w_start:w_end].sum()
        if (center_energy / total) < 0.45: center_fails += 1
            
    return {"background": center_fails >= 2, "spread": False, "border": False}