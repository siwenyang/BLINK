import argparse
import json
from tqdm import tqdm
# import logging
from blink.biencoder.biencoder import load_biencoder
from blink.biencoder.data_process import get_candidate_representation
import torch


parser = argparse.ArgumentParser()

# biencoder
parser.add_argument(
    "--biencoder_model",
    dest="biencoder_model",
    type=str,
    default="models/biencoder_wiki_large.bin",
    help="path to the biencoder model.",
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
    default="models/entity.jsonl",
    help="Path to the entity catalogue.",
)

# output folder
parser.add_argument(
    "--output_path",
    dest="output_path",
    type=str,
    default="models/real_all_river_entities_large.t7",
    help="Path to the output.",
)


args = parser.parse_args()

# logger = utils.get_logger(args.output_path)



# Load biencoder model and biencoder params just like in main_dense.py
with open(args.biencoder_config) as json_file:
    biencoder_params = json.load(json_file)
    biencoder_params["biencoder_model"] = args.biencoder_model
biencoder = load_biencoder(biencoder_params)
print('finish loading biencoder')

# Read 10 entities from entity.jsonl
entities = []
count = 10
with open(args.entity_catalogue) as f:
    for i, line in enumerate(f):
        entity = json.loads(line)
        entities.append(entity)
        if i == count-1:
            break

# Get token_ids corresponding to candidate title and description
print('start generate tokens')
tokenizer = biencoder.tokenizer
max_context_length, max_cand_length =  biencoder_params["max_context_length"], biencoder_params["max_cand_length"]
max_seq_length = max_cand_length
ids = []

for i, entity in enumerate(tqdm(entities)):
    candidate_desc = entity['text']
    candidate_title = entity['title']
    cand_tokens = get_candidate_representation(
        candidate_desc, 
        tokenizer, 
        max_seq_length, 
        candidate_title=candidate_title
    )

    token_ids = cand_tokens["ids"]
    ids.append(token_ids)

    print(candidate_title)
    print(candidate_desc)
    print(cand_tokens)

# print('saving ....')
# ids = torch.tensor(ids)
# torch.save(ids, args.output_path)