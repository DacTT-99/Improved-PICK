import argparse
import torch
from tqdm import tqdm
from pathlib import Path

from torch.utils.data.dataloader import DataLoader
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans

from parse_config import ConfigParser
import model.pick as pick_arch_module
from data_utils.pick_dataset import PICKDataset
from data_utils.pick_dataset import BatchCollateFn
from utils.util import iob_index_to_str, text_index_to_str

def eval(args):
	device = torch.device(f'cuda:{args.gpu}' if args.gpu != -1 else 'cpu')
	checkpoint = torch.load(args.checkpoint, map_location=device)

	config = checkpoint['config']
	state_dict = checkpoint['state_dict']
	monitor_best = checkpoint['monitor_best']
	print('Loading checkpoint: {} \nwith saved mEF {:.4f} ...'.format(args.checkpoint, monitor_best))

	# prepare model for eval
	pick_model = config.init_obj('model_arch', pick_arch_module)
	pick_model = pick_model.to(device)
	pick_model.load_state_dict(state_dict)
	pick_model.eval()

	# setup dataset and data_loader instances
	test_dataset = PICKDataset(	files_name=args.fn,
								boxes_and_transcripts_folder='boxes_and_transcripts',
								images_folder='images',
								entities_folder='entities',
								iob_tagging_type='box_and_within_box_level',
								ignore_error=False,
								training=True)
	test_data_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
									num_workers=2, collate_fn=BatchCollateFn(training=True))

	#setup out path
	output_path = Path(args.output_folder)
	output_path.mkdir(parents=True, exist_ok=True)

	# caculate evaluation meansure
	with torch.no_grad():
		for step_idx, input_data_item in tqdm(enumerate(test_data_loader)):
			for key, input_value in input_data_item.items():
				if input_value is not None and isinstance(input_value, torch.Tensor):
					input_data_item[key] = input_value.to(device)

			# For easier debug.
			image_names = input_data_item["filenames"]

			output = pick_model(**input_data_item)
			logits = output['logits']  # (B, N*T, out_dim)
			new_mask = output['new_mask']
			image_indexs = input_data_item['image_indexs']  # (B,)
			text_segments = input_data_item['text_segments']  # (B, num_boxes, T)
			mask = input_data_item['mask']
			gt_tags = input_data_item['iob_tags_label']

			best_paths = pick_model.decoder.crf_layer.viterbi_tags(logits, mask=new_mask, logits_batch_first=True)
			predicted_tags = []
			for path, score in best_paths:
				predicted_tags.append(path)
		for gt_tag,pred_tag in enumerate(zip(gt_tags,predicted_tags)):
			if gt_tag == pred_tag:
				print('a')



if __name__ == '__main__':
	args = argparse.ArgumentParser(description='PyTorch PICK Testing')
	args.add_argument('-ckpt', '--checkpoint', default=None, type=str,
						help='path to load checkpoint (default: None)')
	args.add_argument('--fn', '--file_name', default=None, type=str,
						help='ocr results folder including boxes and transcripts (default: None)')
	args.add_argument('-output', '--output_folder', default='predict_results', type=str,
						help='output folder (default: predict_results)')
	args.add_argument('-g', '--gpu', default=-1, type=int,
						help='GPU id to use. (default: -1, cpu)')
	args.add_argument('--bs', '--batch_size', default=1, type=int,
						help='batch size (default: 1)')
	args = args.parse_args()
	eval(args)