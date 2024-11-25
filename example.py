import torch
from torchvision.transforms import v2

from run_and_analyse import train_and_compare_models, plot_comparison, evaluate_models
from datasets import get_video_dataloaders
transform = v2.Compose([
    v2.Resize((64, 64)),
    v2.RandomHorizontalFlip(),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #ImageNet normalization
])

#Example script
root_dir = './ufc10'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Train Simple3DCNN
train_loader, val_loader, test_loader = get_video_dataloaders(root_dir, transform, batch_size=8)

# Training with memory tracking
results, models = train_and_compare_models('./ufc10', num_epochs=40)
plot_comparison(results)
evaluate_models(models, test_loader, device)