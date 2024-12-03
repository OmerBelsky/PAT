import sys
import os

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(curdir)

import time
import pandas as pd
import numpy as np
from transformers import BertTokenizerFast, AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModel
from bert_ranker.dataloader.dataset import MSMARCO_PR_Pair_Dataset
import bert_ranker_utils
import metrics
import torch
import torch.nn as nn
import argparse
from apex import amp
from models.bert_cat import BERT_Cat
from models.pointwise_rg import RGRanker
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser('Pytorch')
    # Input and output configs
    parser.add_argument("--output_dir", default=curdir + '/results', type=str,
                        help="the folder to output predictions")
    parser.add_argument("--mode", default='dl2019', type=str,
                        help="eval_full_dev1000/eval_pseudo_full_dev1000/dl2019/eval_subsmall_dev")

    # Training procedure
    parser.add_argument("--seed", default=42, type=str,
                        help="random seed")

    parser.add_argument("--val_batch_size", default=256, type=int,
                        help="Validation and test batch size.")

    # Model hyperparameters
    parser.add_argument("--transformer_model", default="bert-large-uncased", type=str, required=False,
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

    args = parser.parse_args()
    args.model_name = 'pub-ranker'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    now_time = '_'.join(time.asctime(time.localtime(time.time())).split()[:3])
    args.run_id = args.transformer_model + '.public.bert.msmarco.' + now_time
    output_dir = curdir + '/results'
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.transformer_model == 'bert-large-uncased':
        tokenizer = BertTokenizerFast.from_pretrained(args.transformer_model)
        model = AutoModelForSequenceClassification.from_pretrained("nboost/pt-bert-large-msmarco")
    elif args.transformer_model == 'bert-base-uncased':
        tokenizer = BertTokenizerFast.from_pretrained(args.transformer_model)
        model = AutoModelForSequenceClassification.from_pretrained("nboost/pt-bert-base-uncased-msmarco")
    elif args.transformer_model == 'pt-tinybert-msmarco':
        tokenizer = AutoTokenizer.from_pretrained("nboost/pt-tinybert-msmarco")
        model = AutoModelForSequenceClassification.from_pretrained("nboost/pt-tinybert-msmarco")
    elif args.transformer_model == 'distilbert-cat-margin_mse-T2-msmarco':
        tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased")  # honestly not sure if that is the best way to go, but it works :)
        model = BERT_Cat.from_pretrained("sebastian-hofstaetter/distilbert-cat-margin_mse-T2-msmarco")
    elif args.transformer_model == 'condenser':
        tokenizer = AutoTokenizer.from_pretrained("Luyu/condenser")
        model = AutoModel.from_pretrained('Luyu/condenser')
    elif args.transformer_model == 'ms-marco-MiniLM-L-12-v2':
        tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
        model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")

    data_obj = MSMARCO_PR_Pair_Dataset(tokenizer=tokenizer)
    if args.mode.startswith('rank_vicuna_') or args.mode.startswith('rank_zephyr_'):
        del model
        from rank_llm.data import Query, Candidate, Request
        from rank_llm.rerank.listwise import vicuna_reranker, zephyr_reranker

        collection_df = pd.read_csv(prodir + '/data/msmarco_passage/collection.tsv', sep='\t', header=None,
                                    names=['pid', 'passage'])
        collection_df.set_index('pid', inplace=True)
        if args.mode.endswith('dl2019'):
            queries_df = pd.read_csv(prodir + '/data/trec_dl_2019/queries.eval.tsv', sep='\t', header=None, names=['qid', 'query'])
            queries_df.set_index('qid', inplace=True)
        if args.mode.endswith('eval_full_dev1000'):
            queries_df = pd.read_csv(prodir + '/data/trec_dl_2019/queries.dev.tsv', sep='\t', header=None, names=['qid', 'query'])
            queries_df.set_index('qid', inplace=True)
        llm_mode = 'rank_vicuna_' if 'vicuna' in args.mode else 'rank_zephyr_'
        mode = args.mode.split(llm_mode)[1]
        qids = []
        pids = []
        for _, _, tmp_qids, tmp_pids in data_obj.data_generator_mono_dev(mode=mode,
                                                                                               batch_size=args.val_batch_size,
                                                                                               max_seq_len=args.max_seq_len):
            qids += tmp_qids
            pids += tmp_pids
        del data_obj
        del tokenizer
        torch.cuda.empty_cache()
        
        if llm_mode == 'rank_vicuna_':
            rr_llm = vicuna_reranker.VicunaReranker(num_gpus=2)
        else:
            rr_llm = zephyr_reranker.ZephyrReranker(num_gpus=2)
        pairs_df = pd.DataFrame({'qid': qids, 'pid': pids})
        final_df = []
        for qid in pairs_df['qid'].unique():
            query = queries_df.loc[qid, 'query']
            # get all pid, passage pairs for this query
            hits = []
            for _, row in pairs_df[pairs_df['qid'] == qid].iterrows():
                pid = row['pid']
                passage = collection_df.loc[pid, 'passage']
                hits.append((pid, passage, 0.0))
            q = Query(query, qid)
            candidates = [Candidate(docid, score, {'contents': doc}) for docid, doc, score in hits]
            r = Request(query=q, candidates=candidates)
            result = rr_llm.rerank_batch(requests=[r], rank_end=len(candidates))[0]
            for i, candidate in enumerate(result.candidates):
                final_df.append((qid, candidate.docid, i + 1, candidate.score))
        final_df = pd.DataFrame(final_df, columns=['qid', 'pid', 'rank', 'score'])
        pairs_df = pairs_df.merge(final_df, on=['qid', 'pid'])
        pairs_df['Q0'] = 'Q0'
        pairs_df['runid'] = 'Rank_Vicuna' if llm_mode == 'rank_vicuna_' else 'Rank_Zephyr'
        df_save_path = output_dir + '/runs/runs.' + args.run_id + '.' + args.mode + '.csv'
        pairs_df[['qid', 'Q0', 'pid', 'rank', 'score', 'runid']].sort_values(['qid', 'rank']).to_csv(df_save_path, sep='\t', index=False, header=False)

        # take only the top 100 per qid and save to seperate file
        top_100 = pairs_df[pairs_df['rank'] <= 100]
        df_save_path = output_dir + "/run." + args.run_id + '.' + args.mode + ".csv"
        top_100[['qid', 'pid', 'rank']].sort_values(['qid', 'rank']).to_csv(df_save_path, sep='\t', index=False, header=False)
        return
    elif args.mode.startswith('rg_'):
        del model
        collection_df = pd.read_csv(prodir + '/data/msmarco_passage/collection.tsv', sep='\t', header=None,
                                    names=['pid', 'passage'])
        collection_df.set_index('pid', inplace=True)
        if args.mode.endswith('dl2019'):
            queries_df = pd.read_csv(prodir + '/data/trec_dl_2019/queries.eval.tsv', sep='\t', header=None, names=['qid', 'query'])
            queries_df.set_index('qid', inplace=True)
        if args.mode.endswith('eval_full_dev1000'):
            queries_df = pd.read_csv(prodir + '/data/trec_dl_2019/queries.dev.tsv', sep='\t', header=None, names=['qid', 'query'])
            queries_df.set_index('qid', inplace=True)
        llm_mode = 'rg_'
        mode = args.mode.split(llm_mode)[1]
        qids = []
        pids = []
        for _, _, tmp_qids, tmp_pids in data_obj.data_generator_mono_dev(mode=mode,
                                                                                               batch_size=args.val_batch_size,
                                                                                               max_seq_len=args.max_seq_len):
            qids += tmp_qids
            pids += tmp_pids
        del data_obj
        del tokenizer
        torch.cuda.empty_cache()
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", torch_dtype=torch.bfloat16).to(device)
        rr_llm = RGRanker(tokenizer, model)
        pairs_df = pd.DataFrame({'qid': qids, 'pid': pids})
        final_df = []
        for qid in tqdm(pairs_df['qid'].unique()):
            query = queries_df.loc[qid, 'query']
            # get all pid, passage pairs for this query
            hits = []
            for _, row in pairs_df[pairs_df['qid'] == qid].iterrows():
                pid = row['pid']
                passage = collection_df.loc[pid, 'passage']
                hits.append((pid, passage, 0.0))
            result = rr_llm.rerank(hits, query)
            for i, (docid, _, score) in enumerate(result):
                final_df.append((qid, docid, i + 1, score))
        final_df = pd.DataFrame(final_df, columns=['qid', 'pid', 'rank', 'score'])
        pairs_df = pairs_df.merge(final_df, on=['qid', 'pid'])
        pairs_df['Q0'] = 'Q0'
        pairs_df['runid'] = 'Relevance_Generation'
        df_save_path = output_dir + '/runs/runs.' + args.run_id + '.' + args.mode + '.csv'
        pairs_df[['qid', 'Q0', 'pid', 'rank', 'score', 'runid']].sort_values(['qid', 'rank']).to_csv(df_save_path, sep='\t', index=False, header=False)

        # take only the top 100 per qid and save to seperate file
        top_100 = pairs_df[pairs_df['rank'] <= 100]
        df_save_path = output_dir + "/run." + args.run_id + '.' + args.mode + ".csv"
        top_100[['qid', 'pid', 'rank']].sort_values(['qid', 'rank']).to_csv(df_save_path, sep='\t', index=False, header=False)
        return



    model.to(device)
    model = amp.initialize(model, opt_level='O1')
    num_gpu = torch.cuda.device_count()
    if num_gpu > 1:
        devices = [v for v in range(num_gpu)]
        model = nn.DataParallel(model, device_ids=devices)

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
            if args.transformer_model == 'distilbert-cat-margin_mse-T2-msmarco':
                input_ids = batch_encoding['input_ids'].to(device)
                attention_mask = batch_encoding['attention_mask'].to(device)
                scores = model(input_ids=input_ids, attention_mask=attention_mask).squeeze()

                all_logits += scores.tolist()
                all_softmax_logits += scores.tolist()
            elif args.transformer_model == 'ms-marco-MiniLM-L-12-v2':
                outputs = model(**batch_encoding)

                scores = outputs.logits.squeeze()
                all_logits += scores.tolist()
                all_softmax_logits += scores.tolist()
            elif args.transformer_model == 'condenser':
                outputs = model(**batch_encoding)
                scores = outputs.logits[:, 1]
                all_logits += scores.tolist()
                all_softmax_logits += scores.tolist()
            else:
                input_ids = batch_encoding['input_ids'].to(device)
                token_type_ids = batch_encoding['token_type_ids'].to(device)
                attention_mask = batch_encoding['attention_mask'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                logits = outputs[0]

                all_logits += logits[:, 1].tolist()
                all_softmax_logits += torch.softmax(logits, dim=1)[:, 1].tolist()

            # for ms-marco-MiniLM-L-12-v2
            all_flat_labels += tmp_labels
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
            run_df.to_csv(output_dir + "/run." + args.run_id + '.' + args.mode + ".csv", sep='\t', index=False,
                          header=False)

            # For TREC eval
            runs_list = []
            for scores, qids, pids in zip(all_logits, all_qids, all_pids):
                sorted_idx = np.array(scores).argsort()[::-1]
                sorted_scores = np.array(scores)[sorted_idx]
                sorted_qids = np.array(qids)[sorted_idx]
                sorted_pids = np.array(pids)[sorted_idx]
                for rank, (t_qid, t_pid, t_score) in enumerate(zip(sorted_qids, sorted_pids, sorted_scores)):
                    runs_list.append((t_qid, 'Q0', t_pid, rank + 1, t_score, 'BERT-Point'))
            runs_df = pd.DataFrame(runs_list, columns=["qid", "Q0", "pid", "rank", "score", "runid"])
            df_save_path = output_dir + '/runs/runs.' + args.run_id + '.' + args.mode + '.csv'
            os.makedirs(output_dir + '/runs', exist_ok=True)
            runs_df.to_csv(df_save_path, sep='\t', index=False,
                           header=False)


if __name__ == "__main__":
    main()
