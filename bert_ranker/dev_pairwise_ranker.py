import sys
import os

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(curdir)


import time
import pandas as pd
import numpy as np
from transformers import BertTokenizerFast
from bert_ranker.models import pairwise_bert
from bert_ranker.dataloader.dataset import MSMARCO_PR_Pair_Dataset
import bert_ranker_utils
import metrics
import torch
import torch.nn as nn
import argparse
from apex import amp


def main():
    parser = argparse.ArgumentParser('Pytorch')
    # Input and output configs
    parser.add_argument("--output_dir", default=curdir + '/results', type=str,
                        help="the folder to output predictions")
    parser.add_argument("--mode", default='dl2019_imitation', type=str,
                        help="eval_full_dev1000_imitation/dl2019/dl2019_imitation/dl2019_200q/mb2014_imitation")

    # Training procedure
    parser.add_argument("--seed", default=42, type=str,
                        help="random seed")

    parser.add_argument("--val_batch_size", default=1024, type=int,
                        help="Validation and test batch size.")

    # Model hyperparameters
    parser.add_argument("--transformer_model", default="bert-base-uncased", type=str, required=False,
                        help="Bert model to use (default = bert-base-cased).")
    parser.add_argument("--max_seq_len", default=256, type=int, required=False,
                        help="Maximum sequence length for the inputs.")
    parser.add_argument("--lr", default=1e-6, type=float, required=False,
                        help="Learning rate.")
    parser.add_argument("--max_grad_norm", default=1, type=float, required=False,
                        help="Max gradient normalization.")
    parser.add_argument("--accumulation_steps", default=1, type=float, required=False,
                        help="gradient accumulation.")
    parser.add_argument("--warmup_portion", default=0.1, type=float, required=False,
                        help="warmup portion.")
    parser.add_argument("--loss_function", default="label-smoothing-cross-entropy", type=str, required=False,
                        help="Loss function (default is 'cross-entropy').")
    parser.add_argument("--smoothing", default=0.1, type=float, required=False,
                        help="Smoothing hyperparameter used only if loss_function is label-smoothing-cross-entropy.")
    parser.add_argument("--sample_config", default="top_25_last_4", required=True)
    parser.add_argument("--imitate_model_name", default="MiniLM", required=True)
    parser.add_argument("--zero_or_warm", default="further.nq", required=True)

    args = parser.parse_args()
    args.model_name = 'pairwise-BERT-ranker'
    output_dir = curdir + '/results'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = BertTokenizerFast.from_pretrained(args.transformer_model)

    # Instantiate transformer model to be used
    model = pairwise_bert.BertForPairwiseLearning.from_pretrained(args.transformer_model,
                                                                  loss_function=args.loss_function,
                                                                  smoothing=args.smoothing)

    data_obj = MSMARCO_PR_Pair_Dataset(tokenizer=tokenizer)

    if args.mode in ['eval_pseudo_full_dev1000', 'eval_pseudo_subsmall', 'pseudo_pseudo', 'dl2019_pseudo',
                     'mb2014_pseudo']:
        model_path = curdir + '/saved_models/' + model.__class__.__name__ + '.pseudo.' + args.transformer_model + '.pth'
    elif args.mode in ['eval_full_dev1000_same_pseudo', 'dl2019_same_pseudo', 'mb2014_same_pseudo']:
        model_path = curdir + '/saved_models/' + model.__class__.__name__ + '.pseudo.same.' + args.transformer_model + '.pth'
    elif args.mode in ['eval_full_dev1000_imitation', 'dl2019_imitation', 'dl2019_200q', 'mb2014_imitation',
                       'eval_subsmall_imitation']:
        # zero_or_warm = 'further.nq'
        # zero_or_warm = 'further'
        # zero_or_warm = 'straight'
        model_path = curdir + '/saved_models/Imitation.' + args.imitate_model_name + '.' + args.zero_or_warm + '.' + model.__class__.__name__ + '.' + args.sample_config + '.' + args.transformer_model + '.pth'

    else:
        model_path = curdir + '/saved_models/' + model.__class__.__name__ + '.' + args.transformer_model + '.pth'

    model.to(device)
    # model = amp.initialize(model, opt_level='O1')
    num_gpu = torch.cuda.device_count()
    if num_gpu > 1:
        devices = [v for v in range(num_gpu)]
        model = nn.DataParallel(model, device_ids=devices)
        model.module.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path))
    print("load {}".format(model_path))

    with torch.no_grad():
        model.eval()

        all_logits = []
        all_flat_labels = []
        all_softmax_logits = []
        all_qids = []
        all_pids = []
        cnt = 0

        for batch_encoding, tmp_labels, tmp_qids, tmp_pids in data_obj.data_generator_mono_dev(mode=args.mode,
                                                                                               batch_size=args.val_batch_size,
                                                                                               max_seq_len=args.max_seq_len):
            cnt += 1

            pos_input_ids = batch_encoding['input_ids'].to(device)
            pos_token_type_ids = batch_encoding['token_type_ids'].to(device)
            pos_attention_mask = batch_encoding['attention_mask'].to(device)
            neg_input_ids = batch_encoding['input_ids'].to(device)
            neg_token_type_ids = batch_encoding['token_type_ids'].to(device)
            neg_attention_mask = batch_encoding['attention_mask'].to(device)
            true_labels = tmp_labels.to(device)
            outputs = model(
                input_ids_pos=pos_input_ids,
                attention_mask_pos=pos_attention_mask,
                token_type_ids_pos=pos_token_type_ids,
                input_ids_neg=neg_input_ids,
                attention_mask_neg=neg_attention_mask,
                token_type_ids_neg=neg_token_type_ids,
            )
            logits = outputs[0]

            all_flat_labels += true_labels.int().tolist()  # this is required because of the weak supervision
            all_logits += logits[:, 1].tolist()
            all_softmax_logits += torch.softmax(logits, dim=1)[:, 1].tolist()
            all_qids += tmp_qids
            all_pids += tmp_pids

        # accumulates per query
        all_labels, _ = bert_ranker_utils.accumulate_list_by_qid(all_flat_labels, all_qids)
        all_logits, _ = bert_ranker_utils.accumulate_list_by_qid(all_logits, all_qids)
        all_softmax_logits, _ = bert_ranker_utils.accumulate_list_by_qid(all_softmax_logits, all_qids)
        all_pids, all_qids = bert_ranker_utils.accumulate_list_by_qid(all_pids, all_qids)

        res = metrics.evaluate_and_aggregate(all_logits, all_labels, ['ndcg_cut_10', 'map', 'recip_rank'])
        for metric, v in res.items():
            print("\n{} {} : {:3f}".format(args.mode, metric, v))

        validation_metric = ['MAP', 'RPrec', 'MRR', 'MRR@10', 'NDCG', 'NDCG@10']
        all_metrics = np.zeros(len(validation_metric))
        query_cnt = 0
        for labels, logits, probs in zip(all_labels, all_logits, all_softmax_logits):
            #
            gt = set(list(np.where(np.array(labels) > 0)[0]))
            pred_docs = np.array(probs).argsort()[::-1]

            all_metrics += metrics.metrics(gt, pred_docs, validation_metric)
            query_cnt += 1
        all_metrics /= query_cnt
        print("\n" + "\t".join(validation_metric))
        print("\t".join(["{:4f}".format(x) for x in all_metrics]))

        if args.mode not in ['dev', 'test']:
            # Saving predictions and labels to a file
            # For MSMARCO
            top_k = 100
            run_list = []
            for probs, qids, pids in zip(all_logits, all_qids, all_pids):
                sorted_idx = np.array(probs).argsort()[::-1]
                top_qids = np.array(qids)[sorted_idx[:top_k]]
                top_pids = np.array(pids)[sorted_idx[:top_k]]
                for rank, (t_qid, t_pid) in enumerate(zip(top_qids, top_pids)):
                    run_list.append((t_qid, t_pid, rank + 1))
            run_df = pd.DataFrame(run_list, columns=["qid", "pid", "rank"])
            run_df.to_csv(
                output_dir + "/run." + args.mode + '_' + args.imitate_model_name + '_' + args.zero_or_warm + '.' + args.sample_config + '.' + str(
                    args.max_seq_len) + ".csv", sep='\t', index=False, header=False)

            # For TREC eval
            runs_list = []
            for scores, qids, pids in zip(all_logits, all_qids, all_pids):
                sorted_idx = np.array(scores).argsort()[::-1]
                sorted_scores = np.array(scores)[sorted_idx]
                sorted_qids = np.array(qids)[sorted_idx]
                sorted_pids = np.array(pids)[sorted_idx]
                for rank, (t_qid, t_pid, t_score) in enumerate(zip(sorted_qids, sorted_pids, sorted_scores)):
                    runs_list.append((t_qid, 'Q0', t_pid, rank + 1, t_score, 'BERT-Pair'))
            runs_df = pd.DataFrame(runs_list, columns=["qid", "Q0", "pid", "rank", "score", "runid"])
            runs_df.to_csv(
                output_dir + '/runs/runs.' + args.mode + '_' + args.imitate_model_name + '_' + args.zero_or_warm + '.' + args.sample_config + '.' + str(
                    args.max_seq_len) + '.csv', sep='\t', index=False, header=False)


if __name__ == "__main__":
    main()