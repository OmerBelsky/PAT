"""
@Reference: https://github.com/csong27/collision-bert/blob/master/collision_retrieval.py
"""

import sys
import os

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(curdir)

import json
import torch
import logging
import time

import numpy as np
import argparse
import bisect
from tqdm import tqdm
from torch import cuda
import pickle
from transformers import BertTokenizerFast, BertForNextSentencePrediction, \
    AutoModelForSequenceClassification
from transformers import AutoTokenizer
from bert_ranker.models import pairwise_bert, pointwise_rg
from bert_ranker.models.bert_lm import BertForLM
from apex import amp
from attack_methods import pairwise_anchor_trigger
from data_utils import prepare_data_and_scores, pick_target_query_doc_and_best_scores
from rank_llm.data import Query, Candidate, Request
from rank_llm.rerank.listwise import vicuna_reranker, zephyr_reranker
from transformers import T5Tokenizer, T5ForConditionalGeneration

device = 'cuda' if cuda.is_available() else 'cpu'


def main():
    parser = argparse.ArgumentParser('IR_Attack')

    parser.add_argument('--mode', default='test', type=str,
                        help='train/test')

    parser.add_argument("--target", type=str, default='mini', help='test on what model')
    parser.add_argument("--imitation_model", default='imitate.v2', type=str)
    parser.add_argument("--data_name", default="dl", type=str)
    parser.add_argument("--transformer_model", default="bert-base-uncased", type=str, required=False,
                        help="Bert model to use (default = bert-base-cased).")
    parser.add_argument("--tri_len", default=6, type=int, help="Maximun trigger length for generation.")
    parser.add_argument('--min_len', type=int, default=5, help='Min sequence length')
    parser.add_argument("--topk", default=128, type=int, help="Top k sampling for beam search")
    parser.add_argument('--max_iter', type=int, default=20, help='maximum iteraiton')
    parser.add_argument("--lambda_1", default=0.1, type=float, help="Coefficient for language model loss.")
    parser.add_argument('--stemp', type=float, default=0.1, help='temperature of softmax')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='penalty of repetition')
    parser.add_argument('--lr', type=float, default=0.001, help='optimization step size')
    parser.add_argument("--num_beams", default=10, type=int, help="Number of beams")
    parser.add_argument("--num_sim", default=300, type=int, help="Number of similar words")
    parser.add_argument('--perturb_iter', type=int, default=5, help='PPLM iteration')
    parser.add_argument("--seed", default=42, type=str, help="random seed")
    parser.add_argument("--nsp", action='store_true', default=False)
    parser.add_argument("--lambda_2", default=0.8, type=float, help="Coefficient for language model loss.")
    parser.add_argument("--attack_setting_prada", action='store_true', default=False)

    # Support setting
    parser.add_argument("--lm_model_dir", default=prodir + '/data/wiki103/bert', type=str,
                        help="Path to pre-trained language model")

    args = parser.parse_args()

    print('Runing with configurations: {}'.format(json.dumps(args.__dict__, indent=4)))

    tokenizer = BertTokenizerFast.from_pretrained(args.transformer_model)
    if args.mode == 'train':
        train_trigger(args, tokenizer)
    elif args.mode == 'test':
        test_transfer(args)
    else:
        raise ValueError('Not implemented error!')


def train_trigger(args, tokenizer):
    target_q_passage, query_scores, best_query_sent, queries, passages_dict = prepare_data_and_scores(
        target_name=args.target,
        data_name=args.data_name,
        top_k=5,
        least_num=5)
    model = pairwise_bert.BertForPairwiseLearning.from_pretrained(args.transformer_model, torch_dtype=torch.bfloat16)
    model.to(device)
    if args.imitation_model == 'imitate.v2':
        model_path = prodir + '/bert_ranker/saved_models/Imitation.MiniLM.further.BertForPairwiseLearning.top_25_last_4.bert-base-uncased.pth'
    elif args.imitation_model == 'imitate.v1':
        model_path = prodir + '/bert_ranker/saved_models/Imitation.bert_large.further.nq.BertForPairwiseLearning.top_25_last_4.bert-base-uncased.pth'
    elif args.imitation_model == 'pairwise':
        model_path = prodir + '/bert_ranker/saved_models/BertForPairwiseLearning.bert-base-uncased.pth'
    elif args.imitation_model == 'rank_vicuna':
        _, _, _, all_qid_pid_dict = pick_target_query_doc_and_best_scores(args.target, args.data_name, 5, 5)
        model_path = prodir + '/bert_ranker/saved_models/Imitation.rank_vicuna.further.BertForPairwiseLearning.top_25_last_4.bert-base-uncased.pth'
        r_llm = vicuna_reranker.VicunaReranker()
    elif args.imitation_model == 'rank_zephyr':
        _, _, _, all_qid_pid_dict = pick_target_query_doc_and_best_scores(args.target, args.data_name, 5, 5)
        model_path = prodir + '/bert_ranker/saved_models/Imitation.rank_zephyr.further.BertForPairwiseLearning.top_25_last_4.bert-base-uncased.pth'
        r_llm = zephyr_reranker.ZephyrReranker()
    elif args.imitation_model == 'rg':
        _, _, _, all_qid_pid_dict = pick_target_query_doc_and_best_scores(args.target, args.data_name, 5, 5)
        model_path = prodir + '/bert_ranker/saved_models/Imitation.rg.further.BertForPairwiseLearning.top_25_last_4.bert-base-uncased.pth'
        r_llm = pointwise_rg.RGRanker(T5Tokenizer.from_pretrained("google/flan-t5-xxl"), T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", torch_dtype=torch.bfloat16).to(device))
    else:
        model_path = None
    print("Load model from: {}".format(model_path))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    lm_model = BertForLM.from_pretrained(args.lm_model_dir, torch_dtype=torch.bfloat16)
    lm_model.to(device)
    lm_model.eval()
    for param in lm_model.parameters():
        param.requires_grad = False

    if args.nsp:
        nsp_model = BertForNextSentencePrediction.from_pretrained(args.transformer_model, torch_dtype=torch.bfloat16)
        nsp_model.to(device)
        nsp_model.eval()
        for param in nsp_model.parameters():
            param.requires_grad = False
        # model, lm_model, nsp_model = amp.initialize([model, lm_model, nsp_model])
    else:
        model, lm_model = amp.initialize([model, lm_model])
        nsp_model = None

    total_docs_cnt = 0
    success_cnt = 0
    less_500_cnt = 0
    less_100_cnt = 0
    less_50_cnt = 0
    less_20_cnt = 0
    less_10_cnt = 0
    cnt = 0
    boost_rank_list = []
    q_candi_trigger_dict = dict()
    
    if args.attack_setting_prada:
        metrics_file = 'prada_metrics_so_far_{}.txt'.format(args.imitation_model)
    else:
        metrics_file = 'metrics_so_far_{}.txt'.format(args.imitation_model)
    if not os.path.exists(metrics_file):
        with open(metrics_file, 'w') as f:
            f.write('')
    with open(metrics_file, 'r') as f:
        import re
        pattern = r"Query id=(\d+), Doc id=(\d+), old score=([-\d.]+), new score=([-\d.]+), old rank=(\d+), new rank=(\d+)"
        for line in f:
            match = re.search(pattern, line)
            if match:
                # Extract variables from the matched groups
                old_score = float(match.group(3))
                new_score = float(match.group(4))
                old_rank = int(match.group(5))
                new_rank = int(match.group(6))
                total_docs_cnt += 1
                boost_rank_list.append(old_rank - new_rank)
                if old_rank > new_rank:
                    success_cnt += 1
                    if new_rank <= 500:
                        less_500_cnt += 1
                        if new_rank <= 100:
                            less_100_cnt += 1
                            if new_rank <= 50:
                                less_50_cnt += 1
                                if new_rank <= 20:
                                    less_20_cnt += 1
                                    if new_rank <= 10:
                                        less_10_cnt += 1

    used_qids = list(queries.keys())
    qid_did_tracker = {qid: [] for qid in used_qids}
    if args.attack_setting_prada:
        tracker_file = f'prada_qid_did_checkpoint_{args.imitation_model}.pkl' 
    else:
        tracker_file = f'qid_did_checkpoint_{args.imitation_model}.pkl'
    if os.path.exists(tracker_file):
        with open(tracker_file, 'rb') as fin:
            qid_did_tracker = pickle.load(fin)
    for qid in tqdm(used_qids, desc="Processing"):
        torch.manual_seed(args.seed + cnt)
        torch.cuda.manual_seed_all(args.seed + cnt)
        cnt += 1
        tmp_trigger_dict = {}
        query = queries[qid]
        best = best_query_sent[qid]
        best_score = best[0]
        anchor = ' '.join(best[1:4])
        old_scores = query_scores[qid][::-1]
        if int(qid) not in qid_did_tracker:
            qid_did_tracker[int(qid)] = []

        if args.attack_setting_prada and (args.imitation_model in ['rank_vicuna', 'rank_zephyr', 'rg']):
            attacked_docs = dict()
            old_ranks = dict()
            for did in target_q_passage[qid]:
                if (int(qid) in qid_did_tracker) and (int(did) in qid_did_tracker[int(qid)]):
                    continue
                raw_passage = passages_dict[did]
                trigger, new_score, trigger_cands = pairwise_anchor_trigger(
                    query=query,
                    anchor=anchor,
                    raw_passage=raw_passage,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    args=args,
                    lm_model=lm_model,
                    nsp_model=nsp_model)

                msg = f'Query={query}\n' \
                    f'Best true sentences={anchor}\n' \
                    f'Best similarity score={best_score}\n' \
                    f'Trigger={trigger}\n' \
                    f'Similarity core={new_score}\n'
                tmp_trigger_dict[did] = trigger
                print(msg)
                old_rank, old_score = target_q_passage[qid][did]
                old_ranks[did] = old_rank
                attacked_docs[did] = trigger + ' ' + passages_dict[did]
            if args.imitation_model in ['rank_vicuna', 'rank_zephyr']:
                q = Query(query, qid)
                candidates = [Candidate(docid, 0, {'contents': attacked_docs[docid]} if docid in attacked_docs.keys() else {'contents': passages_dict[docid]}) for docid in all_qid_pid_dict[qid]]
                request = Request(query=q, candidates=candidates)
                reranked = r_llm.rerank(request=request, rank_end=100)
                new_ranks = {did:[candidate.docid for candidate in reranked[0].candidates].index(did) + 1 for did in attacked_docs.keys()}
            elif args.imitation_model == 'rg':
                hits = [(did, attacked_docs[did] if did in attacked_docs.keys() else passages_dict[did], 0) for did in all_qid_pid_dict[qid]]
                reranked = r_llm.rerank(hits, query)
                new_ranks = {did:[candidate[0] for candidate in reranked].index(did) + 1 for did in attacked_docs.keys()}
            torch.cuda.empty_cache()
            total_docs_cnt += len(attacked_docs)
            boost_rank_list += [old_ranks[did] - new_ranks[did] for did in attacked_docs]
            success_cnt += sum([1 for did in attacked_docs if old_ranks[did] > new_ranks[did]])
            for did in attacked_docs:
                print(f'Query id={qid}, Doc id={did}, '
                        f'old score=0, new score=0, old rank={old_ranks[did]}, new rank={new_ranks[did]}')
                print('\n\n')
                qid_did_tracker[int(qid)].append(int(did))
                with open(tracker_file, 'wb') as fout:
                    pickle.dump(qid_did_tracker, fout)
                with open(metrics_file, 'a') as fout:
                    fout.write(f'Query id={qid}, Doc id={did}, ' f'old score=0, new score=0, old rank={old_ranks[did]}, new rank={new_ranks[did]}\n\n')

        else:
            for did in target_q_passage[qid]:
                if (int(qid) in qid_did_tracker) and (int(did) in qid_did_tracker[int(qid)]):
                    continue
                raw_passage = passages_dict[did]
                trigger, new_score, trigger_cands = pairwise_anchor_trigger(
                    query=query,
                    anchor=anchor,
                    raw_passage=raw_passage,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    args=args,
                    lm_model=lm_model,
                    nsp_model=nsp_model)

                msg = f'Query={query}\n' \
                    f'Best true sentences={anchor}\n' \
                    f'Best similarity score={best_score}\n' \
                    f'Trigger={trigger}\n' \
                    f'Similarity core={new_score}\n'
                tmp_trigger_dict[did] = trigger
                print(msg)

                old_rank, old_score = target_q_passage[qid][did]
                new_rank = len(old_scores) - bisect.bisect_left(old_scores, new_score)
                if args.imitation_model in ['rank_vicuna', 'rank_zephyr']:
                    q = Query(query, qid)
                    candidates = [Candidate(docid, 0, {'contents': trigger + ' ' + passages_dict[docid]} if docid == did else {'contents': passages_dict[docid]}) for docid in all_qid_pid_dict[qid]]
                    request = Request(query=q, candidates=candidates)
                    reranked = r_llm.rerank(request=request, rank_end=len(candidates))
                    new_rank = [candidate.docid for candidate in reranked[0].candidates].index(did) + 1
                    
                    torch.cuda.empty_cache()
                    
                total_docs_cnt += 1
                boost_rank_list.append(old_rank - new_rank)
                if old_rank > new_rank:
                    success_cnt += 1
                    if new_rank <= 500:
                        less_500_cnt += 1
                        if new_rank <= 100:
                            less_100_cnt += 1
                            if new_rank <= 50:
                                less_50_cnt += 1
                                if new_rank <= 20:
                                    less_20_cnt += 1
                                    if new_rank <= 10:
                                        less_10_cnt += 1

                print(f'Query id={qid}, Doc id={did}, '
                    f'old score={old_score:.2f}, new score={new_score:.2f}, old rank={old_rank}, new rank={new_rank}')
                print('\n\n')
                qid_did_tracker[int(qid)].append(int(did))
                with open(tracker_file, 'wb') as fout:
                    pickle.dump(qid_did_tracker, fout)
                with open(metrics_file, 'a') as fout:
                    fout.write(f'Query id={qid}, Doc id={did}, ' f'old score={old_score:.2f}, new score={new_score:.2f}, old rank={old_rank}, new rank={new_rank}\n\n')


        q_candi_trigger_dict[qid] = tmp_trigger_dict

    boost_success_rate = success_cnt / (total_docs_cnt + 0.0) * 100
    less_500_rate = less_500_cnt / (total_docs_cnt + 0.0) * 100
    less_100_rate = less_100_cnt / (total_docs_cnt + 0.0) * 100
    less_50_rate = less_50_cnt / (total_docs_cnt + 0.0) * 100
    less_20_rate = less_20_cnt / (total_docs_cnt + 0.0) * 100
    less_10_rate = less_10_cnt / (total_docs_cnt + 0.0) * 100
    avg_boost_rank = np.average(boost_rank_list)
    res_str = 'Boost Success Rate: {}\n' \
              'Average Boost Rank: {}\n' \
              'less than 500 Rate: {}\n' \
              'less than 100 Rate: {}\n' \
              'less than 50 Rate: {}\n' \
              'less than 20 Rate: {}\n' \
              'less than 10 Rate: {}\n'.format(boost_success_rate, avg_boost_rank, less_500_rate, less_100_rate,
                                               less_50_rate, less_20_rate, less_10_rate)
    print(res_str)

    if args.num_beams == 1:
        trigger_path = curdir + '/saved_results/greedy_triggers_{}_on_{}_{}_{}_{}.json'.format(args.target,
                                                                                               args.imitation_model,
                                                                                               args.lambda_1,
                                                                                               args.nsp,
                                                                                               args.tri_len)
    else:
        trigger_path = curdir + '/saved_results/triggers_{}_on_{}_{}_{}_{}.json'.format(args.target,
                                                                                        args.imitation_model,
                                                                                        args.lambda_1,
                                                                                        args.nsp,
                                                                                        args.tri_len)
    with open(trigger_path, 'w', encoding='utf-8') as fout:
        fout.write(json.dumps(q_candi_trigger_dict, ensure_ascii=False))
        print('Trigger saved!')


def test_transfer(args):
    print('Test all triggers on imitation model...')
    # load victim models results
    target_q_passage, query_scores, best_query_sent, queries, passages_dict = prepare_data_and_scores(
        target_name=args.target,
        data_name='dl',
        top_k=5,
        least_num=5)

    if args.target == 'mini':
        tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
        model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
        model.to(device)
    elif args.target == 'large':
        tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased')
        model = AutoModelForSequenceClassification.from_pretrained("nboost/pt-bert-large-msmarco")
        model.to(device)
    elif args.target == 'imitate.v1':
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        model = pairwise_bert.BertForPairwiseLearning.from_pretrained(args.transformer_model)
        model.to(device)
        model_path = prodir + '/bert_ranker/saved_models/Imitation.bert_large.further.nq.BertForPairwiseLearning.top_25_last_4.bert-base-uncased.pth'
        print("Load model from: {}".format(model_path))
        model.load_state_dict(torch.load(model_path))

    elif args.target == 'imitate.v2':
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        model = pairwise_bert.BertForPairwiseLearning.from_pretrained(args.transformer_model)
        model.to(device)
        model_path = prodir + '/bert_ranker/saved_models/Imitation.MiniLM.further.nq.BertForPairwiseLearning.top_25_last_4.bert-base-uncased.pth'
        print("Load model from: {}".format(model_path))
        model.load_state_dict(torch.load(model_path))
    else:
        model = None
        tokenizer = None

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    if args.num_beams == 1:
        trigger_path = curdir + '/saved_results/greedy_triggers_{}_on_{}_{}_{}_{}.json'.format(args.target,
                                                                                               args.imitation_model,
                                                                                               args.lambda_1,
                                                                                               args.nsp,
                                                                                               args.tri_len)
    else:
        trigger_path = curdir + '/saved_results/triggers_{}_on_{}_{}_{}_{}.json'.format(args.target,
                                                                                        args.imitation_model,
                                                                                        args.lambda_1,
                                                                                        args.nsp,
                                                                                        args.tri_len)
    with open(trigger_path, 'r', encoding='utf-8') as fin:
        q_candi_trigger_dict = json.loads(fin.readline())
        print("load trigger: {}".format(trigger_path))

    total_docs_cnt = 0
    success_cnt = 0
    less_500_cnt = 0
    less_100_cnt = 0
    less_50_cnt = 0
    less_20_cnt = 0
    less_10_cnt = 0
    boost_rank_list = []
    query_keys = list(q_candi_trigger_dict.keys())

    with torch.no_grad():
        for qid in tqdm(query_keys):
            query = queries[qid]
            old_scores = query_scores[qid][::-1]

            triggers = q_candi_trigger_dict[qid]
            for did in target_q_passage[qid]:
                tmp_best_new_score = -1e9
                old_rank, raw_score = target_q_passage[qid][did]
                for t_did, _ in triggers.items():
                    if args.nsp:
                        # front
                        triggered_passage = triggers[did] + ' ' + passages_dict[did]
                        # end
                        # triggered_passage =  passages_dict[did] + ' ' + triggers[t_did]
                        # middle
                        # half_len_passage = int(len(passages_dict[did]) / 2)
                        # triggered_passage = passages_dict[did][:half_len_passage] + ' ' \
                        #                     + triggers[did] + ' ' + passages_dict[did][half_len_passage:]
                    else:
                        triggered_passage = triggers[t_did] + ' ' + passages_dict[did]

                    batch_encoding = tokenizer([[query, triggered_passage]], max_length=512, padding="max_length",
                                               truncation=True, return_tensors='pt')

                    if args.target == 'mini':
                        outputs = model(**(batch_encoding.to(device)))
                        new_score = outputs.logits.squeeze().item()
                    elif args.target == 'large':
                        outputs = model(**(batch_encoding.to(device)))
                        new_score = outputs.logits[0, -1].item()
                    else:
                        input_ids = batch_encoding['input_ids'].to(device)
                        token_type_ids = batch_encoding['token_type_ids'].to(device)
                        attention_mask = batch_encoding['attention_mask'].to(device)

                        outputs = model(input_ids_pos=input_ids,
                                        attention_mask_pos=attention_mask,
                                        token_type_ids_pos=token_type_ids,
                                        input_ids_neg=input_ids,
                                        attention_mask_neg=attention_mask,
                                        token_type_ids_neg=token_type_ids)[0]
                        new_score = outputs[0, 1].item()
                    if tmp_best_new_score < new_score:
                        tmp_best_new_score = new_score
                        print("New high score: {:.4f}".format(new_score))
                # binary search and return the index
                new_rank = len(old_scores) - bisect.bisect_left(old_scores, tmp_best_new_score)

                total_docs_cnt += 1
                boost_rank_list.append(old_rank - new_rank)
                if old_rank > new_rank:
                    success_cnt += 1
                    if new_rank <= 500:
                        less_500_cnt += 1
                        if new_rank <= 100:
                            less_100_cnt += 1
                            if new_rank <= 50:
                                less_50_cnt += 1
                                if new_rank <= 20:
                                    less_20_cnt += 1
                                    if new_rank <= 10:
                                        less_10_cnt += 1

                print(f'Query id={qid}, Doc id={did}, '
                      f'old score={raw_score:.4f}, new score={tmp_best_new_score:.4f}, old rank={old_rank}, new rank={new_rank}')

            print('\n\n')

    boost_success_rate = success_cnt / (total_docs_cnt + 0.0) * 100
    less_500_rate = less_500_cnt / (total_docs_cnt + 0.0) * 100
    less_100_rate = less_100_cnt / (total_docs_cnt + 0.0) * 100
    less_50_rate = less_50_cnt / (total_docs_cnt + 0.0) * 100
    less_20_rate = less_20_cnt / (total_docs_cnt + 0.0) * 100
    less_10_rate = less_10_cnt / (total_docs_cnt + 0.0) * 100
    avg_boost_rank = np.average(boost_rank_list)
    print("load trigger: {}".format(trigger_path))
    print("Imitation: {}; Target: {}".format(args.imitation_model, args.target))
    res_str = 'Boost Success Rate: {}\n' \
              'Average Boost Rank: {}\n' \
              'less than 500 Rate: {}\n' \
              'less than 100 Rate: {}\n' \
              'less than 50 Rate: {}\n' \
              'less than 20 Rate: {}\n' \
              'less than 10 Rate: {}\n'.format(boost_success_rate, avg_boost_rank, less_500_rate, less_100_rate,
                                               less_50_rate, less_20_rate, less_10_rate)
    print(res_str)
    print("Results:\t10\t20\t50\t100\t500\tSucc\tAvg-Boost\n")
    print("Results:\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(less_10_rate, less_20_rate, less_50_rate, less_100_rate,
                                                        less_500_rate, boost_success_rate, avg_boost_rank))


if __name__ == "__main__":
    main()
