import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time
import torch
import torch.multiprocessing as mp
import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


"""

    single_gpu_train.py의 시간 문제를 non_blocking=True를 통해서 cpu가 데이터를 보내고 바로 다음 데이터를 준비해서 시간을 줄임
    pin_memory = cpu가 gpu로 데이터를 보낼 때 고정핀을 놔서 두번에 걸쳐서 데이터를 보내는 걸 한번에 해결
    persistent_worker = worker들을 상시 유지

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

    #작업 그룹 초기화
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_demo(demo_fn, world_size):
    pass


#DistributedSampler 적용하기!
def load_data(batch_size):
    """
    CIFAR-10 데이터 로드
    """
    # distributedsampler 생성 하고 trian_loader에 적용하기
    # 데이터 증강 및 정규화
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

    # 데이터셋 로드
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    # train_dataset이 정의된 다음 생성    
    train_sampler = DistributedSampler(train_dataset)

    # DataLoader 설정
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler = train_sampler,
        num_workers=4,
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
    """
    한 에폭 학습
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
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
    """
    검증 수행
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return test_loss / len(test_loader), 100. * correct / total
def main():
    world_size = 2
    mp.spawn(train_ddp, args = (world_size,), nprocs=world_size, join=True)


#DistributedDataParallel 적용하기!( main = demo_basic())
def train_ddp(rank, world_size):
    """
    메인 학습 함수
    """
    #프로세스 초기화
    ddp_setup(rank, world_size)
    # 디바이스 설정
    device = rank # rank가 곧 gpu id임

    # 하이퍼파라미터 설정
    batch_size = 256
    num_epochs = 100
    learning_rate = 0.1
    
    # 데이터 로드
    train_loader, test_loader, train_sampler = load_data(batch_size)
    
    # 모델 설정
    model = SimpleCNN().to(rank)
    model = DDP(model, device_ids=[rank])
    
    # 손실 함수와 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    # 결과 저장을 위한 리스트
    epoch_times = []
    
    # 학습 루프
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
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
        
    # 평균 에폭 시간 계산
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    if rank == 0:
        print(f'\nAverage epoch time: {avg_epoch_time:.2f} seconds')

    cleanup()
    

if __name__ == "__main__":
    main()