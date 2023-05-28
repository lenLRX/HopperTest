import torch


def main():
    data = torch.rand((24, 512, 512), device=torch.device('cuda:0'))
    torch.nn.functional.softmax(data, dim=-1)

if __name__ == '__main__':
    main()


