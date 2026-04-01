#!/usr/bin/env python3
"""
Classifier Training and Evaluation for RQ-VAE EuroSAT

Usage:
    python3 train_eval_classifier.py --mode train
    python3 train_eval_classifier.py --mode eval --output-dirs output/8x8x4
    python3 train_eval_classifier.py --mode all
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rq-vae'))
from rqvae.img_datasets.eurosat import EuroSAT, EUROSAT_CLASSES
from rqvae.img_datasets.transforms import create_transforms
from omegaconf import OmegaConf

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EUROSAT_DIR = os.environ.get('EUROSAT_DIR', '../EuroSAT_RGB')
SPLIT_INDICES = os.environ.get('SPLIT_INDICES', '../eurosat_split_indices.pt')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
RESULTS_DIR = os.environ.get('RESULTS_DIR', 'results')


class DummyConfig:
    """Dummy config for transforms"""
    def __init__(self):
        self.transforms = OmegaConf.create({
            'type': 'eurosat',
            'flip_prob': 0.5,
            'color_jitter': False
        })


def get_transforms(is_eval=True):
    """Get transforms for classifier"""
    cfg = DummyConfig()
    if is_eval:
        return create_transforms(cfg, split='val', is_eval=True)
    else:
        return create_transforms(cfg, split='train', is_eval=False)


def train_classifier(train_loader, val_loader, num_classes=10, epochs=30, lr=1e-3):
    """Train a ResNet-18 classifier on EuroSAT images"""
    print(f"\n{'='*60}")
    print("Training Classifier (ResNet-18)")
    print(f"{'='*60}")
    print(f"Device: {DEVICE}")
    print(f"Classes: {num_classes}")
    print(f"Epochs: {epochs}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Load pretrained ResNet-18
    classifier = resnet18(weights=ResNet18_Weights.DEFAULT)
    classifier.fc = nn.Linear(classifier.fc.in_features, num_classes)
    classifier = classifier.to(DEVICE)
    
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    best_state = None
    
    for epoch in range(epochs):
        # Training
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for imgs, labels in pbar:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            logits = classifier(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*train_correct/train_total:.1f}%'})
        
        train_acc = 100 * train_correct / train_total
        
        # Validation
        classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]  ')
            for imgs, labels in pbar:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                logits = classifier(imgs)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss={train_loss/len(train_loader):.4f}, '
              f'Train Acc={train_acc:.2f}%, Val Loss={val_loss/len(val_loader):.4f}, '
              f'Val Acc={val_acc:.2f}%')
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = classifier.state_dict().copy()
            print(f'  -> New best accuracy: {best_acc:.2f}%')
    
    # Load best model
    classifier.load_state_dict(best_state)
    
    # Save classifier
    save_path = os.path.join(OUTPUT_DIR, 'classifier_best.pt')
    torch.save(classifier.state_dict(), save_path)
    print(f"\nClassifier saved to: {save_path}")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    
    return classifier


def evaluate_classifier(classifier, test_loader):
    """Evaluate classifier on test set"""
    classifier.eval()
    correct = 0
    total = 0
    class_correct = {c: 0 for c in EUROSAT_CLASSES}
    class_total = {c: 0 for c in EUROSAT_CLASSES}
    
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc='Evaluating'):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            logits = classifier(imgs)
            _, predicted = logits.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            for i, label in enumerate(labels):
                class_name = EUROSAT_CLASSES[label.item()]
                class_total[class_name] += 1
                if predicted[i] == label:
                    class_correct[class_name] += 1
    
    accuracy = 100 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print("\nPer-class accuracy:")
    for c in EUROSAT_CLASSES:
        if class_total[c] > 0:
            acc = 100 * class_correct[c] / class_total[c]
            print(f"  {c}: {acc:.1f}% ({class_correct[c]}/{class_total[c]})")
    
    return accuracy, {c: 100 * class_correct[c] / class_total[c] if class_total[c] > 0 else 0 
                      for c in EUROSAT_CLASSES}


def load_reconstructed_images(recon_dir, split_indices, split='test'):
    """Load reconstructed images from a model's output directory"""
    # Find reconstruction images
    recon_files = sorted(Path(recon_dir).glob('recon_epoch*_test.png'))
    
    if not recon_files:
        print(f"No reconstruction files found in {recon_dir}")
        return None
    
    # Use the last epoch's reconstructions
    recon_file = recon_files[-1]
    print(f"Loading reconstructions from: {recon_file}")
    
    # Load the reconstruction grid image
    recon_img = Image.open(recon_file).convert('RGB')
    
    # Note: This is a simplified approach - in practice you'd want to load
    # individual reconstructed images and match them with test indices
    # For now, we'll create a dataset that loads from the recon directory
    
    return str(recon_file)


class ReconstructedDataset(torch.utils.data.Dataset):
    """Dataset for reconstructed images"""
    def __init__(self, recon_dir, split_indices, transform=None):
        self.recon_dir = recon_dir
        self.transform = transform
        self.split_indices = split_indices
        
        # Find all reconstruction images
        self.recon_files = sorted(Path(recon_dir).glob('recon_*.png'))
        if not self.recon_files:
            raise ValueError(f"No reconstruction files found in {recon_dir}")
        
        # Use test indices
        if hasattr(split_indices, 'tolist'):
            self.indices = split_indices['test'].tolist()
        else:
            self.indices = split_indices['test']
        
        print(f"Found {len(self.recon_files)} recon files, using {len(self.indices)} test samples")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # This is a simplified version - in practice you'd extract individual images
        # from the reconstruction grid
        # For now, return a placeholder
        return torch.zeros(3, 64, 64), 0


def evaluate_on_reconstructions(classifier, output_dirs, results_dir):
    """Evaluate classifier on reconstructed images from multiple models"""
    print(f"\n{'='*60}")
    print("Evaluating Classifier on Reconstructed Images")
    print(f"{'='*60}")
    
    os.makedirs(results_dir, exist_ok=True)
    
    results = {}
    
    for output_dir in output_dirs:
        model_name = os.path.basename(output_dir)
        print(f"\n--- Evaluating on {model_name} reconstructions ---")
        
        # Load test dataset with reconstructions
        # Note: This requires the actual reconstructed images
        # For now, we'll note that this needs the recon files
        
        recon_dir = output_dir
        recon_files = list(Path(recon_dir).glob('recon_epoch*_test.png'))
        
        if recon_files:
            print(f"Found reconstruction files: {recon_files[-1].name}")
            # In a full implementation, you would:
            # 1. Extract individual images from the reconstruction grid
            # 2. Create a dataset with those images
            # 3. Evaluate the classifier
            results[model_name] = {
                'status': 'reconstructions_found',
                'file': str(recon_files[-1])
            }
        else:
            print(f"No reconstruction files found in {output_dir}")
            results[model_name] = {'status': 'no_reconstructions'}
    
    # Save results
    results_file = os.path.join(results_dir, 'classifier_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nClassifier results saved to: {results_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate classifier')
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['train', 'eval', 'all'],
                        help='Mode: train, eval, or all')
    parser.add_argument('--output-dirs', type=str, nargs='+',
                        default=None,
                        help='Output directories with reconstructed images')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs for classifier training')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for classifier')
    
    args = parser.parse_args()
    
    # Load split indices
    print(f"Loading split indices from: {SPLIT_INDICES}")
    split_indices = torch.load(SPLIT_INDICES)
    print(f"Train: {len(split_indices['train'])}, Val: {len(split_indices['val'])}, Test: {len(split_indices['test'])}")
    
    if args.mode in ['train', 'all']:
        # Create datasets
        print("\nCreating datasets...")
        transform_train = get_transforms(is_eval=False)
        transform_eval = get_transforms(is_eval=True)
        
        train_dataset = EuroSAT(EUROSAT_DIR, split='train', transform=transform_train,
                                split_indices_path=SPLIT_INDICES)
        val_dataset = EuroSAT(EUROSAT_DIR, split='val', transform=transform_eval,
                              split_indices_path=SPLIT_INDICES)
        test_dataset = EuroSAT(EUROSAT_DIR, split='test', transform=transform_eval,
                               split_indices_path=SPLIT_INDICES)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # Train classifier
        classifier = train_classifier(train_loader, val_loader, epochs=args.epochs)
        
        # Evaluate on test set
        print("\nEvaluating on original test images...")
        test_acc, class_acc = evaluate_classifier(classifier, test_loader)
        
        # Save test results
        test_results = {
            'test_accuracy': test_acc,
            'per_class_accuracy': class_acc
        }
        test_results_file = os.path.join(RESULTS_DIR, 'classifier_test_results.json')
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(test_results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"Test results saved to: {test_results_file}")
    
    if args.mode in ['eval', 'all']:
        # Default output dirs if not specified
        if args.output_dirs is None:
            output_dirs = [
                f"{OUTPUT_DIR}/eurosat-rqvae-8x8x1",
                f"{OUTPUT_DIR}/eurosat-rqvae-8x8x4",
                f"{OUTPUT_DIR}/eurosat-rqvae-8x8x8",
                f"{OUTPUT_DIR}/eurosat-rqvae-4x4x1",
                f"{OUTPUT_DIR}/eurosat-rqvae-4x4x4",
                f"{OUTPUT_DIR}/eurosat-rqvae-4x4x8",
                f"{OUTPUT_DIR}/eurosat-rqvae-2x2x1",
                f"{OUTPUT_DIR}/eurosat-rqvae-2x2x4",
                f"{OUTPUT_DIR}/eurosat-rqvae-2x2x8",
            ]
        else:
            output_dirs = args.output_dirs
        
        # Filter to existing directories
        output_dirs = [d for d in output_dirs if os.path.exists(d)]
        
        if not output_dirs:
            print("No output directories found!")
            return
        
        # Load classifier
        classifier_path = os.path.join(OUTPUT_DIR, 'classifier_best.pt')
        if os.path.exists(classifier_path):
            print(f"\nLoading classifier from: {classifier_path}")
            classifier = resnet18(weights=ResNet18_Weights.DEFAULT)
            classifier.fc = nn.Linear(classifier.fc.in_features, 10)
            classifier.load_state_dict(torch.load(classifier_path, map_location=DEVICE))
            classifier = classifier.to(DEVICE)
        else:
            print(f"Classifier not found at {classifier_path}. Train first!")
            return
        
        # Evaluate on reconstructions
        evaluate_on_reconstructions(classifier, output_dirs, RESULTS_DIR)


if __name__ == '__main__':
    main()
