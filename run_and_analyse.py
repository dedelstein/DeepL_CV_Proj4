import torch
import json
import psutil
import time
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from cnn3d import Simple3DCNN, resnet18_3d
from datasets import get_video_dataloaders

def get_memory_usage():
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
        gpu_cached = torch.cuda.memory_reserved() / 1024**2
        return {
            'gpu_allocated': gpu_memory,
            'gpu_cached': gpu_cached,
            'ram': psutil.Process().memory_info().rss / 1024**2
        }
    return {'ram': psutil.Process().memory_info().rss / 1024**2}

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    memory_stats = []
    times_per_epoch = []
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        batch_times = []
        for inputs, labels in train_loader:
            batch_start = time.time()
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            batch_times.append(time.time() - batch_start)
        
        # Calculate metrics
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Track memory and time
        memory_stats.append(get_memory_usage())
        epoch_time = time.time() - epoch_start
        times_per_epoch.append({
            'epoch_time': epoch_time,
            'avg_batch_time': np.mean(batch_times),
            'std_batch_time': np.std(batch_times)
        })
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Time: {epoch_time:.2f}s, Batch avg: {np.mean(batch_times):.3f}s')
        print(f'Memory: {memory_stats[-1]}\n')
    
    return {
        'metrics': (train_losses, train_accs, val_losses, val_accs),
        'memory': memory_stats,
        'time': times_per_epoch
    }

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    batch_times = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            start_time = time.time()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            batch_times.append(time.time() - start_time)
    
    test_acc = 100. * correct / total
    memory_usage = get_memory_usage()
    
    print(f'Test Accuracy: {test_acc:.2f}%')
    print(f'Average inference time per batch: {np.mean(batch_times):.3f}s')
    print(f'Memory usage during inference: {memory_usage}')
    
    return {
        'accuracy': test_acc,
        'batch_times': batch_times,
        'memory': memory_usage
    }

def train_and_compare_models(root_dir, num_epochs=30, batch_size=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = get_video_dataloaders(root_dir, batch_size=batch_size)
    
    # Train both models
    models = {
        'Simple3DCNN': Simple3DCNN(in_channels=3, n_classes=10).to(device),
        'ResNet3D': resnet18_3d(num_classes=10).to(device)
    }
    
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        results[name] = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
        results[name]['test'] = evaluate_model(model, test_loader, device)
    
    return results, models

def plot_comparison(results):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Colors for each model
    colors = {'Simple3DCNN': 'blue', 'ResNet3D': 'red'}
    
    # Plot training and validation loss
    for name, res in results.items():
        train_loss, train_acc, val_loss, val_acc = res['metrics']
        ax1.plot(train_loss, label=f'{name} Train', linestyle='-', color=colors[name])
        ax1.plot(val_loss, label=f'{name} Val', linestyle='--', color=colors[name])
    ax1.set_title('Loss Comparison')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    
    # Plot accuracy
    for name, res in results.items():
        train_loss, train_acc, val_loss, val_acc = res['metrics']
        ax2.plot(train_acc, label=f'{name} Train', linestyle='-', color=colors[name])
        ax2.plot(val_acc, label=f'{name} Val', linestyle='--', color=colors[name])
    ax2.set_title('Accuracy Comparison')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    # Plot memory usage
    for name, res in results.items():
        if torch.cuda.is_available():
            ax3.plot([m['gpu_allocated'] for m in res['memory']], 
                    label=f'{name} GPU', color=colors[name])
    ax3.set_title('GPU Memory Usage (MB)')
    ax3.set_xlabel('Epoch')
    ax3.legend()
    
    # Plot timing
    for name, res in results.items():
        times = res['time']
        ax4.plot([t['epoch_time'] for t in times], 
                label=f'{name} Epoch Time', color=colors[name])
    ax4.set_title('Training Time (s)')
    ax4.set_xlabel('Epoch')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png')
    plt.show()

    # Save training time
    total_times = {name: sum(t['epoch_time'] for t in res['time']) 
                  for name, res in results.items()}
    
    # Save results as JSON
    results_dict = {
        'training_times': total_times,
        'metrics': {name: {'train_acc': res['metrics'][1][-1],
                          'val_acc': res['metrics'][3][-1]} 
                   for name, res in results.items()}
    }
    with open('training_results.json', 'w') as f:
        json.dump(results_dict, f)
    
    # Print test results
    print("\nTest Results:")
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"Accuracy: {res['test']['accuracy']:.2f}%")
        print(f"Avg inference time: {np.mean(res['test']['batch_times']):.3f}s")
        print(f"Memory usage: {res['test']['memory']}")

def evaluate_models(models, test_loader, device):
    for name, model in models.items():
        model.eval()
        all_preds = []
        all_labels = []
        example_images = []
        example_preds = []
        example_labels = []
        
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = outputs.max(1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                if i == 0:
                    example_images = inputs.cpu()
                    example_preds = preds.cpu().numpy()
                    example_labels = labels.cpu().numpy()
        
        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center')
                
        plt.title(f'{name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'{name}_confusion.png')
        plt.show()
        
        # Accuracy
        accuracy = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds)
        print(f'\n{name} Test Accuracy: {accuracy:.2f}%')
        
        # Predictions Visualization
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        axes = axes.ravel()
        
        for idx in range(8):
            mid_frame = example_images[idx, :, 5].permute(1, 2, 0)
            axes[idx].imshow(mid_frame)
            color = 'green' if example_preds[idx] == example_labels[idx] else 'red'
            axes[idx].set_title(f'Pred: {example_preds[idx]}\nTrue: {example_labels[idx]}', 
                              color=color)
            axes[idx].axis('off')
            
        plt.suptitle(f'{name} Predictions')
        plt.tight_layout()
        plt.savefig(f'{name}_predictions.png')
        plt.show()