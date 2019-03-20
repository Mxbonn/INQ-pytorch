import torch
import torch.nn.functional as F

import src


def train(model, device, loader, optimizer):
    model.train()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()


def test(model, device, loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))

    return test_loss, correct


def save_checkpoint(state, filename):
    torch.save(state, filename)
    print("Saved model to {}".format(filename))


def load_checkpoint(filename):
    print("Loading checkpoint {}".format(filename))
    return torch.load(filename)


def main():
    train_flag = False
    test_flag = False
    debug_flag = True
    epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    train_loader, test_loader = src.data.mnist.get_data_loaders()
    model = src.models.MnistNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)

    if train_flag:
        for epoch in range(1, epochs + 1):
            train(model, device, train_loader, optimizer)
            test(model, device, test_loader)

        save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename="./models/mnist.pth.tar")

    if test_flag:
        checkpoint = load_checkpoint("./models/mnist.pth.tar")
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        test(model, device, test_loader)

    if debug_flag:
        checkpoint = load_checkpoint("./models/mnist.pth.tar")
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        optimizer = src.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)
        scheduler = src.optim.INQScheduler(optimizer, iterative_steps=[0.5, 1])
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)
        test(model, device, test_loader)
        train(model, device, train_loader, optimizer)
        scheduler.step()
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)
                print(param.data.unique().size())
        test(model, device, test_loader)


if __name__ == '__main__':
    main()
