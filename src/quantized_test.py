import torch
import torchvision

from collections import OrderedDict


def main():
    model = torchvision.models.alexnet()
    checkpoint = torch.load("./models/quantized_alexnet.pth.tar")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k.replace('.module', '')
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            print(param.data.unique().size())


if __name__ == '__main__':
    main()
