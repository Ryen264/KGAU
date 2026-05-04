import os
import json
import argparse
import multiprocessing as mp
from multiprocessing import Pool
from typing import List

def _check_sanity(relation_id_to_str: dict):
    relation_str_to_id = {}
    for rel_id, rel_str in relation_id_to_str.items():
        if rel_str is None:
            continue
        if rel_str not in relation_str_to_id:
            relation_str_to_id[rel_str] = rel_id
        elif relation_str_to_id[rel_str] != rel_id:
            assert False, f'ERROR: {relation_str_to_id[rel_str]} and {rel_id} are both normalized to {rel_str}'
    return

def _normalize_relations(examples: List[dict], normalize_fn, is_train: bool, args=None):
    relation_id_to_str = {}
    for ex in examples:
        rel_str = normalize_fn(ex['relation'])
        relation_id_to_str[ex['relation']] = rel_str
        ex['relation'] = rel_str
    _check_sanity(relation_id_to_str)
    if is_train and args is not None:
        out_path = f"{os.path.dirname(args.train_path)}/relations.json"
        with open(out_path, 'w', encoding='utf-8') as writer:
            json.dump(relation_id_to_str, writer, ensure_ascii=False, indent=4)
            print(f'Save {len(relation_id_to_str)} relations to {out_path}')

def _truncate(text: str, max_len: int):
    return ' '.join(text.split()[:max_len])

wn18rr_id2ent = {}
def _load_wn18rr_texts(path: str):
    global wn18rr_id2ent
    lines = open(path, 'r', encoding='utf-8').readlines()
    for line in lines:
        fs = line.strip().split('\t')
        assert len(fs) == 3, f'Invalid line: {line.strip()}'
        entity_id, word, desc = fs[0], fs[1].replace('__', ''), fs[2]
        wn18rr_id2ent[entity_id] = (entity_id, word, desc)
    print(f'Load {len(wn18rr_id2ent)} entities from {path}')

def _process_line_wn18rr(line: str) -> dict:
    fs = line.strip().split('\t')
    assert len(fs) == 3, f'Expect 3 fields for {line}'
    head_id, relation, tail_id = fs[0], fs[1], fs[2]
    _, head, _ = wn18rr_id2ent[head_id]
    _, tail, _ = wn18rr_id2ent[tail_id]
    example = {'head_id': head_id, 'head': head, 'relation': relation, 'tail_id': tail_id, 'tail': tail}
    return example

def preprocess_wn18rr(path, args):
    if not wn18rr_id2ent:
        _load_wn18rr_texts(f"{os.path.dirname(path)}/wordnet-mlj12-definitions.txt")
    lines = open(path, 'r', encoding='utf-8').readlines()
    pool = Pool(processes=args.workers)
    examples = pool.map(_process_line_wn18rr, lines)
    pool.close()
    pool.join()
    _normalize_relations(examples, normalize_fn=lambda rel: rel.replace('_', ' ').strip(), is_train=(path == args.train_path), args=args)
    out_path = path + '.json'
    json.dump(examples, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    print(f'Save {len(examples)} examples to {out_path}')
    return examples

fb15k_id2ent = {}
fb15k_id2desc = {}
def _load_fb15k237_wikidata(path: str):
    global fb15k_id2ent, fb15k_id2desc
    lines = open(path, 'r', encoding='utf-8').readlines()
    for line in lines:
        fs = line.strip().split('\t')
        assert len(fs) == 2, f'Invalid line: {line.strip()}'
        entity_id, name = fs[0], fs[1]
        name = name.replace('_', ' ').strip()
        if entity_id not in fb15k_id2desc:
            print(f'No desc found for {entity_id}')
        fb15k_id2ent[entity_id] = (entity_id, name, fb15k_id2desc.get(entity_id, ''))
    print(f'Load {len(fb15k_id2ent)} entity names from {path}')

def _load_fb15k237_desc(path: str):
    global fb15k_id2desc
    lines = open(path, 'r', encoding='utf-8').readlines()
    for line in lines:
        fs = line.strip().split('\t')
        assert len(fs) == 2, f'Invalid line: {line.strip()}'
        entity_id, desc = fs[0], fs[1]
        fb15k_id2desc[entity_id] = _truncate(desc, 50)
    print(f'Load {len(fb15k_id2desc)} entity descriptions from {path}')

def _normalize_fb15k237_relation(relation: str) -> str:
    tokens = relation.replace('./', '/').replace('_', ' ').strip().split('/')
    dedup_tokens = []
    for token in tokens:
        if token not in dedup_tokens[-3:]:
            dedup_tokens.append(token)
    relation_tokens = dedup_tokens[::-1]
    relation = ' '.join([t for idx, t in enumerate(relation_tokens) if idx == 0 or relation_tokens[idx] != relation_tokens[idx - 1]])
    return relation

def _process_line_fb15k237(line: str) -> dict:
    fs = line.strip().split('\t')
    assert len(fs) == 3, f'Expect 3 fields for {line}'
    head_id, relation, tail_id = fs[0], fs[1], fs[2]
    _, head, _ = fb15k_id2ent[head_id]
    _, tail, _ = fb15k_id2ent[tail_id]
    example = {'head_id': head_id, 'head': head, 'relation': relation, 'tail_id': tail_id, 'tail': tail}
    return example

def preprocess_fb15k237(path, args):
    if not fb15k_id2desc:
        _load_fb15k237_desc(f"{os.path.dirname(path)}/FB15k_mid2description.txt")
    if not fb15k_id2ent:
        _load_fb15k237_wikidata(f"{os.path.dirname(path)}/FB15k_mid2name.txt")
    lines = open(path, 'r', encoding='utf-8').readlines()
    pool = Pool(processes=args.workers)
    examples = pool.map(_process_line_fb15k237, lines)
    pool.close()
    pool.join()
    _normalize_relations(examples, normalize_fn=_normalize_fb15k237_relation, is_train=(path == args.train_path), args=args)
    out_path = path + '.json'
    json.dump(examples, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    print(f'Save {len(examples)} examples to {out_path}')
    return examples

def dump_all_entities(examples, out_path, id2text: dict):
    id2entity = {}
    relations = set()
    for ex in examples:
        head_id = ex['head_id']
        relations.add(ex['relation'])
        if head_id not in id2entity:
            id2entity[head_id] = {'entity_id': head_id, 'entity': ex['head'], 'entity_desc': id2text.get(head_id, '')}
        tail_id = ex['tail_id']
        if tail_id not in id2entity:
            id2entity[tail_id] = {'entity_id': tail_id, 'entity': ex['tail'], 'entity_desc': id2text.get(tail_id, '')}
    print(f'Get {len(id2entity)} entities, {len(relations)} relations in total')
    json.dump(list(id2entity.values()), open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

def main():
    parser = argparse.ArgumentParser(description='preprocess')
    parser.add_argument('--task', default='wn18rr', type=str, metavar='N', help='dataset name')
    parser.add_argument('--workers', default=2, type=int, metavar='N', help='number of workers')
    parser.add_argument('--train-path', default='', type=str, metavar='N', help='path to training data')
    parser.add_argument('--valid-path', default='', type=str, metavar='N', help='path to valid data')
    parser.add_argument('--test-path', default='', type=str, metavar='N', help='path to test data')
    args = parser.parse_args()
    mp.set_start_method('fork', force=True)
    all_examples = []
    for path in [args.train_path, args.valid_path, args.test_path]:
        assert os.path.exists(path), f"File not found: {path}"
        print(f'Process {path}...')
        if args.task.lower() == 'wn18rr':
            all_examples += preprocess_wn18rr(path, args)
        elif args.task.lower() == 'fb15k237':
            all_examples += preprocess_fb15k237(path, args)
        else:
            assert False, f'Unknown task: {args.task}'
    if args.task.lower() == 'wn18rr':
        id2text = {k: v[2] for k, v in wn18rr_id2ent.items()}
    elif args.task.lower() == 'fb15k237':
        id2text = {k: v[2] for k, v in fb15k_id2ent.items()}
    else:
        assert False, f'Unknown task: {args.task}'
    dump_all_entities(all_examples, out_path=f"{os.path.dirname(args.train_path)}/entities.json", id2text=id2text)
    print('Done')

if __name__ == '__main__':
    main()
