import argparse
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torchvision

from model import Net
from tripletfolder import TripletFolder


parser = argparse.ArgumentParser(description="Train on market1501")
parser.add_argument("--data-dir", default='data', type=str)
parser.add_argument("--no-cuda", action="store_true")
parser.add_argument("--gpu-id", default=0, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--interval", '-i', default=20, type=int)
parser.add_argument('--resume', '-r', action='store_true')
args = parser.parse_args()

# device
device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
if torch.cuda.is_available() and not args.no_cuda:
    cudnn.benchmark = True

# data loading
root = args.data_dir
train_dir = os.path.join(root, "train")
test_dir = os.path.join(root, "test")
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 128)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 128)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
trainloader = torch.utils.data.DataLoader(
    TripletFolder(train_dir, transform=transform_train),
    batch_size=64, shuffle=True
)
testloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(test_dir, transform=transform_test),
    batch_size=64, shuffle=True
)
num_classes = len(trainloader.dataset.classes)

# net definition
start_epoch = 0
net = Net(num_classes=num_classes)
if args.resume:
    assert os.path.isfile("./checkpoint/ckpt.t7"), "Error: no checkpoint file found!"
    print('Loading from checkpoint/ckpt.t7')
    checkpoint = torch.load("./checkpoint/ckpt.t7")
    # import ipdb; ipdb.set_trace()
    net_dict = checkpoint['net_dict']
    net.load_state_dict(net_dict)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
net.to(device)

# loss and optimizer
criterion = torch.nn.TripletMarginLoss()
# optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)
optimizer = torch.optim.Adam(net.parameters(), args.lr, weight_decay=5e-4)
best_acc = 0.


# train function for each epoch
def train(epoch):
    print("\nEpoch : %d" % (epoch + 1))
    net.train()
    training_loss = 0.
    train_loss = 0.
    correct = 0
    total = 0
    interval = args.interval
    start = time.time()
    for idx, (inputs, labels, positives, negatives) in enumerate(trainloader):
        # forward
        inputs = inputs.to(device)
        positives = positives.to(device)
        negatives = negatives.to(device)
        labels = labels.to(device)

        outputs, output_features = net(inputs)
        _, positive_features = net(positives)
        _, negative_features = net(negatives)

        loss = criterion(output_features, positive_features, negative_features)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumurating
        training_loss += loss.item()
        train_loss += loss.item()
        correct += outputs.max(dim=1)[1].eq(labels).sum().item()
        total += labels.size(0)

        # print
        if (idx + 1) % interval == 0:
            end = time.time()
            print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100. * (idx + 1) / len(trainloader), end - start, training_loss / interval, correct, total, 100. * correct / total
            ))
            training_loss = 0.
            start = time.time()

    return train_loss / len(trainloader), 1. - correct / total


def test(epoch):
    global best_acc
    net.eval()
    correct, total = 0, 0
    start = time.time()
    with torch.no_grad():
        all_features, all_labels = [], []
        for idx, (inputs, labels) in enumerate(testloader):
            inputs = inputs.to(device)
            _, features = net(inputs)
            all_features.append(features.cpu())
            all_labels.append(labels)
            # correct += outputs.max(dim=1)[1].eq(labels).sum().item()
            total += labels.size(0)
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        scores = all_features.mm(all_features.t())
        scores.masked_fill_(torch.eye(*scores.size()).bool(), 0)
        top1s = scores.topk(5, dim=1)[1][:, 0]
        correct = all_labels[top1s].eq(all_labels).sum().item()

        print("Testing ...")
        end = time.time()
        print("[progress:{:.1f}%]time:{:.2f}s Correct:{}/{} Acc:{:.3f}%".format(
            100. * (idx + 1) / len(testloader), end - start, correct, total, 100. * correct / total
        ))

    # saving checkpoint
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        print("Saving parameters to checkpoint/ckpt.t7")
        checkpoint = {
            'net_dict': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(checkpoint, './checkpoint/ckpt.t7')

    return 1. - correct / total


# plot figure
x_epoch = []
record = {'train_loss': [], 'train_err': [], 'test_loss': [], 'test_err': []}
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


def draw_curve(epoch, train_loss, train_err, test_loss, test_err):
    global record
    record['train_loss'].append(train_loss)
    record['train_err'].append(train_err)
    record['test_loss'].append(test_loss)
    record['test_err'].append(test_err)

    x_epoch.append(epoch)
    ax0.plot(x_epoch, record['train_loss'], 'bo-', label='train')
    ax0.plot(x_epoch, record['test_loss'], 'ro-', label='val')
    ax1.plot(x_epoch, record['train_err'], 'bo-', label='train')
    ax1.plot(x_epoch, record['test_err'], 'ro-', label='val')
    if epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig("train.jpg")


# lr decay
def lr_decay():
    global optimizer
    for params in optimizer.param_groups:
        params['lr'] *= 0.1
        lr = params['lr']
        print("Learning rate adjusted to {}".format(lr))


def main():
    for epoch in range(start_epoch, start_epoch + 20):
        train_loss, train_err = train(epoch)
        test_err = test(epoch)
        draw_curve(epoch, train_loss, train_err, train_loss, test_err)
        if (epoch + 1) % 20 == 0:
            lr_decay()


if __name__ == '__main__':
    main()
