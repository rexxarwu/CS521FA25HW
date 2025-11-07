import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json
import os 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
from skimage.segmentation import slic
from sklearn.linear_model import Ridge
from skimage.transform import resize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the pre-trained ResNet18 model
model = models.resnet18(pretrained=True).to(device)
model.eval()  # Set model to evaluation mode

# Define the image preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]   
    )
])

# Load the ImageNet class index mapping
with open("imagenet_class_index.json") as f:
    class_idx = json.load(f)
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
idx2synset = [class_idx[str(k)][0] for k in range(len(class_idx))]
id2label = {v[0]: v[1] for v in class_idx.values()}

imagenet_path = './imagenet_samples'

# List of image file paths
image_paths = os.listdir(imagenet_path)


for img_path in image_paths:
    # Open and preprocess the image
    my_img = os.path.join(imagenet_path, img_path)
    #my_img = os.path.join(img_path, os.listdir(img_path)[2])
    input_image = Image.open(my_img).convert('RGB')
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0).to(device)  # Create a mini-batch as expected by the model

    # Move the input and model to GPU if available
    # if torch.cuda.is_available():
    #     input_batch = input_batch.to('cuda')
    #     model.to('cuda')

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    # Get the predicted class index
    _, predicted_idx = torch.max(output, 1)
    predicted_idx = predicted_idx.item()
    predicted_synset = idx2synset[predicted_idx]
    predicted_label = idx2label[predicted_idx]

    print(f"Predicted label: {predicted_synset} ({predicted_label})")
    
#Segment image into superpixels
def segment_image(image, num_segments=50):
    img_arr = np.array(image)
    segments = slic(img_arr, n_segments=num_segments, compactness=10, start_label=0)
    return segments

# Perturbed samples
def generate_perturbations(image, segments, num_samples=1000):
    num_features = np.unique(segments).shape[0]
    img_arr = np.array(image)
    samples = np.random.randint(0, 2, size=(num_samples, num_features))
    perturbed_images = []
    for i in range(num_samples):
        mask = samples[i]
        temp = img_arr.copy()
        temp[np.isin(segments, np.where(mask == 0))] = 0
        perturbed_images.append(Image.fromarray(temp))
    return samples, perturbed_images

# Linear model fitting
def get_model_outputs_and_weights(model, original_image, perturbed_images, samples, sigma=25):
    with torch.no_grad():
        orig_pred = model(preprocess(original_image).unsqueeze(0).to(device))
        target_class = orig_pred.argmax().item()

    preds = []
    weights = []
    for i, img in enumerate(perturbed_images):
        x = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.nn.functional.softmax(model(x), dim=1)[0, target_class].item()
        preds.append(prob)
        dist = np.sum(samples[i] == 0)
        weights.append(np.exp(-dist / sigma))
    return np.array(preds), np.array(weights)

# Weighted linear model
def fit_local_surrogate(samples, preds, weights):
    reg = Ridge(alpha=1.0)
    reg.fit(samples, preds, sample_weight=weights)
    importance = np.abs(reg.coef_)
    return importance

def lime_explanation(model, image, num_samples=1000, num_segments=50):
    segments = segment_image(image, num_segments)
    samples, perturbed_images = generate_perturbations(image, segments, num_samples)
    preds, weights = get_model_outputs_and_weights(model, image, perturbed_images, samples)
    importance = fit_local_surrogate(samples, preds, weights)
    return segments, importance

# Visualization
def show_lime_result(image, segments, importance, img_name="output"):
    img_arr = np.array(image)
    importance_norm = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
    num_top = max(1, len(importance) // 5)
    top_segments = np.argsort(importance_norm)[-num_top:]
    mask = np.zeros_like(segments, dtype=bool)
    for seg in top_segments:
        mask |= (segments == seg)
    img_gray = np.dot(img_arr[...,:3], [0.2989, 0.5870, 0.1140])
    img_gray = np.stack([img_gray]*3, axis=-1).astype(np.uint8)
    lime_vis = np.where(mask[...,None], img_arr, img_gray)
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1); plt.imshow(image); plt.title("Original"); plt.axis('off')
    plt.subplot(1,2,2); plt.imshow(lime_vis.astype(np.uint8)); plt.title("LIME"); plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"lime_{img_name}.png", dpi=300)
    plt.close()
    return lime_vis


# SmoothGrad
def smoothgrad(model, image, target_class=None, num_samples=50, noise_sigma=0.15):
    x = preprocess(image).unsqueeze(0).to(device)
    x.requires_grad = True

    if target_class is None:
        with torch.no_grad():
            target_class = model(x).argmax().item()

    grads = []
    for _ in range(num_samples):
        noise = torch.randn_like(x).to(device) * noise_sigma
        x_noisy = x + noise
        logits = model(x_noisy)
        score = logits[0, target_class]
        model.zero_grad()
        score.backward(retain_graph=True)
        grads.append(x.grad.detach().clone())
        x.grad.zero_()

    avg_grad = torch.mean(torch.stack(grads), dim=0).squeeze().permute(1,2,0)
    saliency = avg_grad.abs().mean(dim=-1).cpu().numpy()
    return saliency

def show_smoothgrad_result(image, saliency, img_name="output"):
    img_arr = np.array(image)
    saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    threshold = np.percentile(saliency_norm, 80)
    mask = saliency_norm >= threshold

    mask_resized = resize(mask, (img_arr.shape[0], img_arr.shape[1]), mode='reflect', anti_aliasing=False) > 0.5

    img_gray = np.dot(img_arr[...,:3], [0.2989, 0.5870, 0.1140])
    img_gray = np.stack([img_gray]*3, axis=-1).astype(np.uint8)
    smooth_vis = np.where(mask_resized[..., None], img_arr, img_gray)

    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1); plt.imshow(image); plt.title("Original"); plt.axis('off')
    plt.subplot(1,2,2); plt.imshow(smooth_vis.astype(np.uint8)); plt.title("SmoothGrad"); plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"smooth_{img_name}.png", dpi=300)
    plt.close()

    return smooth_vis

# Correlation comparison
def compare_explanations(lime_map, smooth_map):
    if lime_map.ndim == 3:
        lime_map = lime_map.mean(axis=-1)
    if smooth_map.ndim == 3:
        smooth_map = smooth_map.mean(axis=-1)
    if lime_map.shape != smooth_map.shape:
        lime_map = resize(lime_map, smooth_map.shape, mode='reflect', anti_aliasing=True)

    lime_flat = lime_map.flatten()
    smooth_flat = smooth_map.flatten()
    valid = np.isfinite(lime_flat) & np.isfinite(smooth_flat)
    spearman_corr, _ = spearmanr(lime_flat[valid], smooth_flat[valid])
    kendall_corr, _ = kendalltau(lime_flat[valid], smooth_flat[valid])
    return spearman_corr, kendall_corr


for i, img_name in enumerate(image_paths):
    img_path = os.path.join(imagenet_path, img_name)
    input_image = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    _, predicted_idx = torch.max(output, 1)
    predicted_idx = predicted_idx.item()
    predicted_synset = idx2synset[predicted_idx]
    predicted_label = idx2label[predicted_idx]

    print(f"\nImage {i+1}/{len(image_paths)}: {img_name}")
    print(f"Predicted label: {predicted_synset} ({predicted_label})")

    segments, importance = lime_explanation(model, input_image, num_samples=300, num_segments=50)
    lime_map = show_lime_result(input_image, segments, importance, img_name=f"{i}_{img_name}")
    smooth_map = smoothgrad(model, input_image)
    show_smoothgrad_result(input_image, smooth_map, img_name=f"{i}_{img_name}")

    spearman_corr, kendall_corr = compare_explanations(lime_map, smooth_map)
    print(f"Spearman ρ = {spearman_corr:.4f}, Kendall τ = {kendall_corr:.4f}")
