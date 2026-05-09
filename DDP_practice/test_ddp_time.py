import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler



"""
#============================================================================
# ddp의 시간이 너무 길게 걸리는 것 같아서 어느 부분에서 시간이 걸리는지 확인한 코드
#============================================================================
"""




class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def load_data(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    train_sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_sampler

def train_epoch(model, train_loader, criterion, optimizer, epoch, device, rank):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    train_loader.sampler.set_epoch(epoch)
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # ==================== 시간 측정 코드 (첫 배치에만 적용) ====================
        if batch_idx == 0:
            torch.cuda.synchronize(device) # 정확한 측정을 위한 동기화
            
            # 1. 순전파(Forward Pass) 시간 측정
            start_time = time.perf_counter()
            outputs = model(images)
            torch.cuda.synchronize(device)
            forward_time = time.perf_counter() - start_time

            loss = criterion(outputs, labels)

            # 2. 역전파(Backward Pass, 그래디언트 동기화 포함) 시간 측정
            start_time = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize(device)
            backward_time = time.perf_counter() - start_time

            if rank == 0:
                print("\n" + "="*50)
                print(f"Epoch {epoch} Timing Analysis (Batch 0):")
                print(f"  - Forward Pass Time         : {forward_time:.6f} 초")
                print(f"  - Backward Pass + Sync Time : {backward_time:.6f} 초")
                print(f"  - 비율 (역전파/순전파)      : {backward_time / forward_time:.2f} 배")
                print("="*50 + "\n")
        # ========================================================================
        else: # 원래 학습 코드
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if rank == 0 and batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, '
                  f'Loss: {running_loss/(batch_idx+1):.3f}, '
                  f'Acc: {100.*correct/total:.3f}%')
    
    return running_loss / len(train_loader), 100. * correct / total

def validate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model.module(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return test_loss / len(test_loader), 100. * correct / total

def train_ddp(rank, world_size):
    ddp_setup(rank, world_size)
    device = rank

    batch_size = 256
    num_epochs = 10 # 시간을 절약하기 위해 에폭 수를 줄여서 테스트
    learning_rate = 0.1
    
    train_loader, test_loader, _ = load_data(batch_size)
    
    model = SimpleCNN().to(rank)
    model = DDP(model, device_ids=[rank])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    epoch_times = []
    
    for epoch in range(num_epochs):
        epoch_start_time = time.perf_counter()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch, device, rank)
        
        if rank == 0:
            test_loss, test_acc = validate(model, test_loader, criterion, device)
        
        epoch_end_time = time.perf_counter()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        
        scheduler.step()
        
        if rank == 0:
            print(f'\nEpoch: {epoch}')
            print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}%')
            print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f}%')
            print(f'Epoch Time: {epoch_time:.2f} seconds\n')
            
    if rank == 0:
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        print(f'\nAverage epoch time: {avg_epoch_time:.2f} seconds')

    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)