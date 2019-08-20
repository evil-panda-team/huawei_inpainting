import argparse
import os

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils import data
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from data_vlad import DS
from loss import InpaintingLoss
from model import DFNet


class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='train_shuffled.flist')
parser.add_argument('--save_dir', type=str, default='./snapshots/bboxes')
parser.add_argument('--log_dir', type=str, default='./logs/bboxes')
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--max_iter', type=int, default=680000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--vis_interval', type=int, default=1000)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--lr_scheduler_interval', type=int, default=50000)
parser.add_argument('--image_size', type=int, default=512)
parser.add_argument('--resume', type=str, default='model/model_places2.pth')
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
device = torch.device('cuda')

if not os.path.exists(args.save_dir):
    os.makedirs('{:s}/images'.format(args.save_dir))
    os.makedirs('{:s}/ckpt'.format(args.save_dir))
    
print("SAVEDIRS MADE")

writer = SummaryWriter(logdir=args.log_dir)

print("SUMMARY WRITER MADE")

size = (args.image_size, args.image_size)
print(size)

img_tf = transforms.Compose([
    transforms.Resize(size=size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
    
print("IMAGE TRANSFORMS MADE")

dataset = DS(args.root, img_tf)

print("Loading iterator train")
iterator_train = iter(data.DataLoader(
    dataset, batch_size=args.batch_size,
    sampler=InfiniteSampler(len(dataset)),
    num_workers=args.n_threads
))
print(len(dataset))

model = DFNet().to(device)

lr = args.lr

start_iter = 0
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
criterion = InpaintingLoss().to(device)

if args.resume:
    checkpoint = torch.load(args.resume, map_location=device)
    model.load_state_dict(checkpoint)

for i in tqdm(range(start_iter, args.max_iter)):
    model.train()

    img, mask = [x.to(device) for x in next(iterator_train)]
    masked = img * mask

    results, alpha, raw = model(masked, mask)
    loss = criterion(results, img)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i + 1) % args.log_interval == 0:
        writer.add_scalar('loss', loss.item(), i + 1)

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        torch.save(model.state_dict(), '{:s}/ckpt/{:d}.pth'.format(args.save_dir, i + 1))

    if (i + 1) % args.vis_interval == 0:
        s_img = torch.cat([img, masked, results[0]])
        s_img = make_grid(s_img, nrow=args.batch_size)
        save_image(s_img, '{:s}/images/test_{:d}.png'.format(args.save_dir, i + 1))

    if (i + 1) % args.lr_scheduler_interval:
        scheduler.step()
