# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import json
import sys

from tqdm import tqdm
import logging
import torch
import numpy as np
from colorama import init
from termcolor import colored

import blink.ner as NER
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from blink.biencoder.biencoder import BiEncoderRanker, load_biencoder
from blink.crossencoder.crossencoder import CrossEncoderRanker, load_crossencoder
from blink.biencoder.data_process import (
    process_mention_data,
    get_candidate_representation,
)
import blink.candidate_ranking.utils as utils
from blink.crossencoder.train_cross import modify, evaluate
from blink.crossencoder.data_process import prepare_crossencoder_data
from blink.indexer.faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer
from multiprocessing import Pool, cpu_count
from itertools import repeat
import itertools
import threading
import time
import torch.multiprocessing
from multiprocessing import set_start_method, get_context
# torch.multiprocessing.set_sharing_strategy('file_system')
# print('torch multiprocessing')



HIGHLIGHTS = [
    "on_red",
    "on_green",
    "on_yellow",
    "on_blue",
    "on_magenta",
    "on_cyan",
]


def _print_colorful_text(input_sentence, samples):
    init()  # colorful output
    msg = ""
    if samples and (len(samples) > 0):
        msg += input_sentence[0 : int(samples[0]["start_pos"])]
        for idx, sample in enumerate(samples):
            msg += colored(
                input_sentence[int(sample["start_pos"]) : int(sample["end_pos"])],
                "grey",
                HIGHLIGHTS[idx % len(HIGHLIGHTS)],
            )
            if idx < len(samples) - 1:
                msg += input_sentence[
                    int(sample["end_pos"]) : int(samples[idx + 1]["start_pos"])
                ]
            else:
                msg += input_sentence[int(sample["end_pos"]) :]
    else:
        msg = input_sentence
        print("Failed to identify entity from text:")
    print("\n" + str(msg) + "\n")


def _print_colorful_prediction(
    idx, sample, e_id, e_title, e_text, e_url, show_url=False
):
    print(colored(sample["mention"], "grey", HIGHLIGHTS[idx % len(HIGHLIGHTS)]))
    to_print = "id:{}\ntitle:{}\ntext:{}\n".format(e_id, e_title, e_text[:256])
    if show_url:
        to_print += "url:{}\n".format(e_url)
    print(to_print)


def _annotate(ner_model, input_sentences):
    ner_output_data = ner_model.predict(input_sentences)
    sentences = ner_output_data["sentences"]
    mentions = ner_output_data["mentions"]
    samples = []
    for mention in mentions:
        record = {}
        record["label"] = "unknown"
        record["label_id"] = -1
        # LOWERCASE EVERYTHING !
        record["context_left"] = sentences[mention["sent_idx"]][
            : mention["start_pos"]
        ].lower()
        record["context_right"] = sentences[mention["sent_idx"]][
            mention["end_pos"] :
        ].lower()
        record["mention"] = mention["text"].lower()
        record["start_pos"] = int(mention["start_pos"])
        record["end_pos"] = int(mention["end_pos"])
        record["sent_idx"] = mention["sent_idx"]
        samples.append(record)
    return samples


def _load_candidates(
    entity_catalogue, entity_encoding, faiss_index=None, index_path=None, logger=None
):
    # only load candidate encoding if not using faiss index
    if faiss_index is None:
        candidate_encoding = torch.load(entity_encoding)
        indexer = None
    else:
        if logger:
            logger.info("Using faiss index to retrieve entities.")
        candidate_encoding = None
        assert index_path is not None, "Error! Empty indexer path."
        if faiss_index == "flat":
            indexer = DenseFlatIndexer(1)
        elif faiss_index == "hnsw":
            indexer = DenseHNSWFlatIndexer(1)
        else:
            raise ValueError("Error! Unsupported indexer type! Choose from flat,hnsw.")
        indexer.deserialize_from(index_path)

    # load all the 5903527 entities
    title2id = {}
    id2title = {}
    id2text = {}
    wikipedia_id2local_id = {}
    local_idx = 0
    with open(entity_catalogue, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            entity = json.loads(line)

            if "idx" in entity:
                split = entity["idx"].split("curid=")
                if len(split) > 1:
                    wikipedia_id = int(split[-1].strip())
                else:
                    wikipedia_id = entity["idx"].strip()

                assert wikipedia_id not in wikipedia_id2local_id
                wikipedia_id2local_id[wikipedia_id] = local_idx

            title2id[entity["title"]] = local_idx
            id2title[local_idx] = entity["title"]
            id2text[local_idx] = entity["text"]
            local_idx += 1
    return (
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        indexer,
    )


def __map_test_entities(test_entities_path, title2id, logger):
    # load the 732859 tac_kbp_ref_know_base entities
    kb2id = {}
    missing_pages = 0
    n = 0
    with open(test_entities_path, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            entity = json.loads(line)
            if entity["title"] not in title2id:
                missing_pages += 1
            else:
                kb2id[entity["entity_id"]] = title2id[entity["title"]]
            n += 1
    if logger:
        logger.info("missing {}/{} pages".format(missing_pages, n))
    return kb2id


def __load_test(test_filename, kb2id, wikipedia_id2local_id, logger):
    test_samples = []
    with open(test_filename, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            record = json.loads(line)
            record["label"] = str(record["label_id"])

            # for tac kbp we should use a separate knowledge source to get the entity id (label_id)
            if kb2id and len(kb2id) > 0:
                if record["label"] in kb2id:
                    record["label_id"] = kb2id[record["label"]]
                else:
                    continue

            # check that each entity id (label_id) is in the entity collection
            elif wikipedia_id2local_id and len(wikipedia_id2local_id) > 0:
                try:
                    key = int(record["label"].strip())
                    if key in wikipedia_id2local_id:
                        record["label_id"] = wikipedia_id2local_id[key]
                    else:
                        continue
                except:
                    continue

            # LOWERCASE EVERYTHING !
            record["context_left"] = record["context_left"].lower()
            record["context_right"] = record["context_right"].lower()
            record["mention"] = record["mention"].lower()
            test_samples.append(record)

    if logger:
        logger.info("{}/{} samples considered".format(len(test_samples), len(lines)))
    return test_samples


def _get_test_samples(
    test_filename, test_entities_path, title2id, wikipedia_id2local_id, logger
):
    kb2id = None
    if test_entities_path:
        kb2id = __map_test_entities(test_entities_path, title2id, logger)
    test_samples = __load_test(test_filename, kb2id, wikipedia_id2local_id, logger)
    return test_samples


def _process_biencoder_dataloader(samples, tokenizer, biencoder_params):
    _, tensor_data = process_mention_data(
        samples,
        tokenizer,
        biencoder_params["max_context_length"],
        biencoder_params["max_cand_length"],
        silent=True,
        logger=None,
        debug=biencoder_params["debug"],
    )
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=biencoder_params["eval_batch_size"]
    )
    return dataloader


def _run_biencoder(biencoder, dataloader, candidate_encoding, top_k=100, indexer=None):
    biencoder.model.eval()
    labels = []
    nns = []
    all_scores = []
    for batch in tqdm(dataloader):
        context_input, _, label_ids = batch
        with torch.no_grad():
            if indexer is not None:
                context_encoding = biencoder.encode_context(context_input).numpy()
                context_encoding = np.ascontiguousarray(context_encoding)
                scores, indicies = indexer.search_knn(context_encoding, top_k)
            else:
                scores = biencoder.score_candidate(
                    context_input, None, cand_encs=candidate_encoding  # .to(device)
                )
                scores, indicies = scores.topk(top_k)
                scores = scores.data.numpy()
                indicies = indicies.data.numpy()

        labels.extend(label_ids.data.numpy())
        nns.extend(indicies)
        all_scores.extend(scores)
    return labels, nns, all_scores


def _process_crossencoder_dataloader(context_input, label_input, crossencoder_params):
    tensor_data = TensorDataset(context_input, label_input)
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=crossencoder_params["eval_batch_size"]
    )
    return dataloader


def _run_crossencoder(crossencoder, dataloader, logger, context_len, device="cuda"):
    crossencoder.model.eval()
    accuracy = 0.0
    crossencoder.to(device)

    res = evaluate(crossencoder, dataloader, device, logger, context_len, zeshel=False, silent=False)
    accuracy = res["normalized_accuracy"]
    logits = res["logits"]

    if accuracy > -1:
        predictions = np.argsort(logits, axis=1)
    else:
        predictions = []

    return accuracy, predictions, logits


def load_models(args, logger=None):

    # load biencoder model
    if logger:
        logger.info("loading biencoder model")
    with open(args.biencoder_config) as json_file:
        biencoder_params = json.load(json_file)
        biencoder_params["path_to_model"] = args.biencoder_model
    biencoder = load_biencoder(biencoder_params)

    crossencoder = None
    crossencoder_params = None
    if not args.fast:
        # load crossencoder model
        if logger:
            logger.info("loading crossencoder model")
        with open(args.crossencoder_config) as json_file:
            crossencoder_params = json.load(json_file)
            crossencoder_params["path_to_model"] = args.crossencoder_model
        crossencoder = load_crossencoder(crossencoder_params)

    # load candidate entities
    if logger:
        logger.info("loading candidate entities")
    (
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        faiss_indexer,
    ) = _load_candidates(
        args.entity_catalogue, 
        args.entity_encoding, 
        faiss_index=getattr(args, 'faiss_index', None), 
        index_path=getattr(args, 'index_path' , None),
        logger=logger,
    )

    return (
        biencoder,
        biencoder_params,
        crossencoder,
        crossencoder_params,
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        faiss_indexer,
    )


# multiprocessing, start here
def main(m_args):
    logger = utils.get_logger(m_args.output_path)
    models = load_models(m_args, logger)
    biencoder = models[0]
    biencoder_params = models[1]
    candidate_encoding = models[4]
    id2title = models[6]
    id2text = models[7]
    wikipedia_id2local_id = models[8]
    id2url = {
        v: "https://en.wikipedia.org/wiki?curid=%s" % k
        for k, v in wikipedia_id2local_id.items()
    }
    top_k = m_args.top_k
    batch_run(m_args, logger, biencoder, biencoder_params, candidate_encoding, id2title, id2text, id2url, top_k)


def identiy_mentions(entry_list, m_args, logger):
    models = load_models(m_args, logger)
    ner_model = NER.get_model()
    biencoder = models[0]
    biencoder_params = models[1]
    candidate_encoding = models[4]
    id2title = models[6]
    id2text = models[7]
    wikipedia_id2local_id = models[8]
    id2url = {
        v: "https://en.wikipedia.org/wiki?curid=%s" % k
        for k, v in wikipedia_id2local_id.items()
    }

    # with open(save_path, 'w') as s_f:
    res_lst = []
    for entry in entry_list:
        text_id = entry[0]
        text = entry[1]
        samples = _annotate(ner_model, [text])

        # prepare the data for biencoder
        dataloader = _process_biencoder_dataloader(
            samples, biencoder.tokenizer, biencoder_params
        )

        # run biencoder
        labels, nns, scores = _run_biencoder(
            biencoder, dataloader, candidate_encoding, m_args.top_k, None
        )

        # print biencoder prediction
        idx = 0
        e_id_lst = []
        e_title_lst = []
        for entity_list, sample in zip(nns, samples):
            e_id = entity_list[0]
            e_title = id2title[e_id]
            e_text = id2text[e_id]
            # e_url = id2url[e_id]
            e_id_lst.append(e_id)
            e_title_lst.append(e_title)
            idx += 1
            # write_content += str(e_id) + '\t'
            # write_content += str(e_title) + '\t'
            # write_content += e_text + '\t'
        # s_f.write(write_content + '\n')
        res_lst.append([text_id, e_id_lst, e_title_lst])
    return res_lst


def batch_run(
    args,
    logger,
    biencoder,
    biencoder_params,
    candidate_encoding,
    id2title,
    id2text,
    id2url,
    top_k,
    multi_cpus_num=1,
):
    # id2url = {
    #     v: "https://en.wikipedia.org/wiki?curid=%s" % k
    #     for k, v in wikipedia_id2local_id.items()
    # }

    samples = None
    total_ms_passage = []
    samples_list = []

    # # Load NER model
    m_ner_model = NER.get_model()

    # load text
    # ms_passage_path = 'models/docs00.json'
    ms_passage_path = args.ms_passage_path
    with open(ms_passage_path, 'r') as m_f:
        lines = m_f.readlines()
        doc_start_index = args.doc_start_index
        # set bucket size following jobs number
        # bucket_size = 100
        bucket_size = args.bucket_size
        for line in lines[doc_start_index*bucket_size : (doc_start_index*bucket_size)+bucket_size]:
            entry = json.loads(line.strip())
            total_ms_passage.append((entry['id'], entry['contents']))


    # Identify mentions
    save_path = 'data/ms_entities_' + str(doc_start_index) + '.txt'
    if multi_cpus_num == 1:
        t1 = time.time()
        result_lst = []
        for entry in tqdm(total_ms_passage):
            text = entry[1]
            text_id = entry[0]
            samples = _annotate(m_ner_model, [text])

            # prepare the data for biencoder
            dataloader = _process_biencoder_dataloader(
                samples, biencoder.tokenizer, biencoder_params
            )

            # run biencoder
            labels, nns, scores = _run_biencoder(
                biencoder, dataloader, candidate_encoding, top_k, None
            )

            # print biencoder prediction
            idx = 0
            e_id_lst, e_title_lst, e_idx_lst = [], [], []
            entities_res = []
            for entity_list, sample in zip(nns, samples):
                temp_res = {}
                e_id = entity_list[0]
                e_title = id2title[e_id]
                e_text = id2text[e_id]
                # e_url = id2url[e_id]
                e_idx = [sample['start_pos'], sample['end_pos']]
                e_id_lst.append(e_id)
                e_title_lst.append(e_title)
                e_idx_lst.append(e_idx)
                # print(e_title)
                # print(e_text)
                idx += 1
                temp_res['id'] = e_id
                temp_res['start_pos'] = sample['start_pos']
                temp_res['end_pos'] = sample['end_pos']
                temp_res['text'] = e_title
                entities_res.append(temp_res)
                # write_content += str(e_id) + '\t'
                # write_content += str(e_title) + '\t'
                # write_content += e_text + '\t's
            # s_f.write(write_content + '\n')   
            # ent_lst.append([e_id_lst, e_title_lst])
            # text_lst.append(text_id)
            res_dict = {}
            res_dict['doc_id'] = text_id
            res_dict['entities'] = entities_res
            result_lst.append(res_dict)
            # result_lst.append([text_id, e_id_lst, e_title_lst, e_idx_lst])

        with open(save_path, 'w') as s_f:
            for result in result_lst:
                content = json.dumps(result) + '\n'
                s_f.write(content)
        t2 = time.time()
        print('use' ,str(t2-t1), 's')
  
    else:
        # single thread
        t1 = time.time()
        # result_lst = identiy_mentions(total_ms_passage)

        # use pool, not work
        with get_context("spawn").Pool(multi_cpus_num) as p:
            print('create pools')
            queries_list = []
            buck_size = int(len(total_ms_passage)/multi_cpus_num)
            for i in range(multi_cpus_num):
                start_index = i*buck_size
                if i == multi_cpus_num-1:
                    queries_list.append(total_ms_passage[start_index:])

                else:
                    queries_list.append(total_ms_passage[start_index:start_index+buck_size])
            aargs = [args]
            llogger = [logger]
            result_lst = p.starmap(identiy_mentions, itertools.product(queries_list, aargs, llogger))
            print('pooldone')

        t2 = time.time()
        print('use' ,str(t2-t1), 's')
        # print('TOTAL reuslts ', str(result_lst))

        with open(save_path, 'w') as s_f:
            for result in result_lst:
                content = str(result[0][0]) + '\t' + str(result[0][1]) + '\t' + str(result[0][2]) + '\n'
                s_f.write(content)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive mode."
    )

    # test_data
    parser.add_argument(
        "--test_mentions", dest="test_mentions", type=str, help="Test Dataset."
    )
    parser.add_argument(
        "--test_entities", dest="test_entities", type=str, help="Test Entities."
    )

    # biencoder
    parser.add_argument(
        "--biencoder_model",
        dest="biencoder_model",
        type=str,
        default="models/biencoder_wiki_large.bin",
        help="Path to the biencoder model.",
    )
    parser.add_argument(
        "--biencoder_config",
        dest="biencoder_config",
        type=str,
        default="models/biencoder_wiki_large.json",
        help="Path to the biencoder configuration.",
    )
    parser.add_argument(
        "--entity_catalogue",
        dest="entity_catalogue",
        type=str,
        # default="models/tac_entity.jsonl",  # TAC-KBP
        default="models/entity.jsonl",  # ALL WIKIPEDIA!
        help="Path to the entity catalogue.",
    )
    parser.add_argument(
        "--entity_encoding",
        dest="entity_encoding",
        type=str,
        # default="models/tac_candidate_encode_large.t7",  # TAC-KBP
        default="models/all_entities_large.t7",  # ALL WIKIPEDIA!
        help="Path to the entity catalogue.",
    )

    # crossencoder
    parser.add_argument(
        "--crossencoder_model",
        dest="crossencoder_model",
        type=str,
        default="models/crossencoder_wiki_large.bin",
        help="Path to the crossencoder model.",
    )
    parser.add_argument(
        "--crossencoder_config",
        dest="crossencoder_config",
        type=str,
        default="models/crossencoder_wiki_large.json",
        help="Path to the crossencoder configuration.",
    )

    parser.add_argument(
        "--top_k",
        dest="top_k",
        type=int,
        default=10,
        help="Number of candidates retrieved by biencoder.",
    )

    # output folder
    parser.add_argument(
        "--output_path",
        dest="output_path",
        type=str,
        default="output",
        help="Path to the output.",
    )

    parser.add_argument(
        "--fast", dest="fast", action="store_true", help="only biencoder mode"
    )

    parser.add_argument(
        "--show_url",
        dest="show_url",
        action="store_true",
        help="whether to show entity url in interactive mode",
    )

    parser.add_argument(
        "--faiss_index", type=str, default=None, help="whether to use faiss index",
    )

    parser.add_argument(
        "--index_path", type=str, default=None, help="path to load indexer",
    )

    parser.add_argument(
        "--doc_start_index", type=int, default=0, help="doc start index",
    )

    parser.add_argument(
        "--bucket_size", type=int, default=0, help="set bucket size following jobs amount",
    )

    parser.add_argument(
        "--ms_passage_path", type=str, default='models/docs00.json', help="the doc to process",
    )

    args = parser.parse_args()
    set_start_method("spawn")
    main(args)