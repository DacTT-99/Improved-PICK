import argparse
import torch
from tqdm import tqdm
from pathlib import Path
import os
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from torch.utils.data.dataloader import DataLoader
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans

from parse_config import ConfigParser
import model.pick as pick_arch_module
from data_utils.pick_dataset import PICKDataset
from data_utils.pick_dataset import BatchCollateFn
from utils.util import iob_index_to_str, text_index_to_str
from utils.class_utils import iob_labels_vocab_cls

def eval(args):
    device = torch.device(f'cuda:{args.gpu}' if args.gpu != -1 else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)

    config = checkpoint['config']
    state_dict = checkpoint['state_dict']
    monitor_best = checkpoint['monitor_best']
    print('Loading checkpoint: {} \nwith saved mEF {:.4f} ...'.format(args.checkpoint, monitor_best))

    # prepare model for eval
    pick_model = config.init_obj(['model_arch'], pick_arch_module)
    pick_model = pick_model.to(device)
    pick_model.load_state_dict(state_dict)
    pick_model.eval()

    # setup dataset and data_loader instances
    test_dataset = PICKDataset(files_name=args.fn,
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

    num_classes = 3
    confusion_matrix = torch.zeros([num_classes+1,num_classes+1])
    
    # caculate evaluation meansure
    with torch.no_grad():
        for _, input_data_item in tqdm(enumerate(test_data_loader)):
            for key, input_value in input_data_item.items():
                if input_value is not None and isinstance(input_value, torch.Tensor):
                    input_data_item[key] = input_value.to(device)

            output = pick_model(**input_data_item)
            logits = output['logits']  # (B, N*T, out_dim)
            new_mask = output['new_mask']

            gt_masks = input_data_item['mask']
            gt_tags = input_data_item['iob_tags_label']
            gt_text_len = input_data_item['text_length']

            best_paths = pick_model.decoder.crf_layer.viterbi_tags(
                logits, mask=new_mask, logits_batch_first=True)
            predicted_tags = []
            for path, _ in best_paths:
                predicted_tags.append(torch.Tensor(path))

            doc_seq_len = gt_text_len.sum(dim=-1)

            B, N, T = gt_tags.shape
            gt_tags = gt_tags.reshape(B, N * T)
            gt_masks = gt_masks.reshape(B, N * T)
            new_gt_tags = torch.zeros_like(gt_tags, dtype=torch.int64)
            new_gt_masks = torch.zeros_like(gt_masks)

            for i in range(B):
                doc_x = gt_tags[i]
                doc_mask_x = gt_masks[i]
                valid_doc_x = doc_x[doc_mask_x == 1]
                num_valid = valid_doc_x.size(0)
                new_gt_tags[i, :num_valid] = valid_doc_x
                new_gt_masks[i, :doc_seq_len[i]] = 1

            new_gt_tags[new_gt_tags < num_classes] += num_classes
            for i in range(B):
                prev = 0
                predicted_tags[i][predicted_tags[i] < num_classes] += num_classes
                for box_len in gt_text_len[i]:
                    if box_len == 0:
                        continue
                    gt_class = torch.argmax(torch.bincount(new_gt_tags[i][prev:prev + box_len].int()))
                    pred_class = torch.argmax(torch.bincount(predicted_tags[i][prev:prev + box_len].int()))
                    confusion_matrix[gt_class - num_classes][pred_class - num_classes] += 1
                    prev += box_len
    confusion_matrix = torch.flip(confusion_matrix, [1])
    tag = [iob_labels_vocab_cls.itos[x].split('-')[1] for x in range(num_classes)]
    tag.append('other')
    df_cm = pd.DataFrame(confusion_matrix.numpy()(),
                         index=[i for i in tag],
                         columns=[i for i in reversed(tag)])
    plt.figure(figsize=(5, 4))
    sn.heatmap(df_cm, annot=True, fmt='g')
    plt.xlabel('predict')
    plt.ylabel('label')
    plt.savefig(os.path.join(args.output_folder,args.fn.split('.')[0] + '.png'))

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