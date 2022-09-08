from argparse import ArgumentParser
import os
import json
import sys
from unittest import result
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from finetune_hybrid_encoder import face_parsing, initFaceParsing

sys.path.append(".")
sys.path.append("..")

from inversion.criteria.lpips.lpips import LPIPS
from inversion.datasets.gt_res_dataset import GTResDataset

class mIOU(torch.nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.bisNet, _ = initFaceParsing(path='pretrained_models', device=None)
	
	def forward(self, source, target):
		
		source = face_parsing(source, self.bisNet)
		target = face_parsing(target, self.bisNet)
		result = torch.mean(torch.div(
			torch.sum(source * target, dim=[2, 3]).float(),
			torch.sum((source + target)>0, dim=[2, 3]).float() + 1e-6), dim=1)
		return result


def parse_args():
	parser = ArgumentParser(add_help=False)
	parser.add_argument('--mode', type=str, default='lpips', choices=['lpips', 'l2', 'miou'])
	parser.add_argument('--output_path', type=str, default='results')
	parser.add_argument('--gt_path', type=str, default='gt_images')
	parser.add_argument('--workers', type=int, default=4)
	parser.add_argument('--batch_size', type=int, default=4)
	args = parser.parse_args()
	return args


def run(args):
	for step in sorted(os.listdir(args.output_path)):
		# if not step.isdigit():
		# 	continue
		# step_outputs_path = os.path.join(args.output_path, step)
		# if os.path.isdir(step_outputs_path):
		# 	print('#' * 80)
		# 	print(f'Running on step: {step}')
		# 	print('#' * 80)
		yaw = float(step.split('_')[-1])
		if yaw < 1.9:
			continue
		output_path = os.path.join(args.output_path, step, 'can')
		gt_path = os.path.join(args.gt_path, step, 'gt')
		run_on_step_output(output_path, gt_path, args=args)


def run_on_step_output(output_path, gt_path, args):

	transform = transforms.Compose([transforms.Resize((256, 256)),
									transforms.ToTensor(),
									transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

	# step_outputs_path = os.path.join(args.output_path, step)
	step_outputs_path = output_path

	print('Loading dataset')
	dataset = GTResDataset(root_path=step_outputs_path,
						   gt_dir=gt_path,
						   transform=transform)

	dataloader = DataLoader(dataset,
							batch_size=args.batch_size,
							shuffle=False,
							num_workers=int(args.workers),
							drop_last=True)

	if args.mode == 'lpips':
		loss_func = LPIPS(net_type='alex')
	elif args.mode == 'l2':
		loss_func = torch.nn.MSELoss()
	elif args.mode == 'miou':
		loss_func = mIOU()
	else:
		raise Exception('Not a valid mode!')
	# if args.mode != 'miou':
	loss_func.cuda()

	global_i = 0
	scores_dict = {}
	all_scores = []
	for result_batch, gt_batch in tqdm(dataloader):
		for i in range(args.batch_size):
			loss = float(loss_func(result_batch[i:i+1].cuda(), gt_batch[i:i+1].cuda()))
			all_scores.append(loss)
			im_path = dataset.pairs[global_i][0]
			scores_dict[os.path.basename(im_path)] = loss
			global_i += 1

	all_scores = list(scores_dict.values())
	mean = np.mean(all_scores)
	std = np.std(all_scores)
	result_str = 'Average loss is {:.2f}+-{:.2f}'.format(mean, std)
	print('Finished with ', step_outputs_path)
	print(result_str)

	out_path = os.path.join(os.path.dirname(output_path), 'inference_metrics')
	if not os.path.exists(out_path):
		os.makedirs(out_path)

	with open(os.path.join(out_path, f'stat_{args.mode}_can.txt'), 'w') as f:
		f.write(result_str)
	with open(os.path.join(out_path, f'scores_{args.mode}_can.json'), 'w') as f:
		json.dump(scores_dict, f)


if __name__ == '__main__':
	args = parse_args()
	run(args)