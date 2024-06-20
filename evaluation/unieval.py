import json
import pickle

from argparse import ArgumentParser

from utils import convert_to_json
from metric.evaluator import get_evaluator


def load_json(filename):
    with open(filename, "r") as file:
        return json.load(file)


def load_jsonl(filename):
    lines = []
    with open(filename, "r") as file:
        for line in file:
            lines.append(json.loads(line))
    return lines


def pickle_dump(data, filename):
    with open(filename, "wb") as file:
        pickle.dump(data, file)

task = 'summarization'


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument("--preds_file", type=str, help="model prediction (generated summaries) file.")
    argparser.add_argument("--data_file", type=str, help="data file for prediction, containing original dialogues.")
    argparser.add_argument("--dlg_column", type=str, default="dialogue")
    argparser.add_argument("--reference_column", type=str, default="summary")
    argparser.add_argument("--output_file", type=str, help="output file to save scores.")
    argparser.add_argument("--dataset_name", type=str, choices=["dialogsum", "samsum"])

    args = argparser.parse_args()

    # read model predictions
    if args.preds_file.endswith(".txt"):
        preds = open(args.preds_file, "r").readlines()
    elif args.preds_file.endswith(".json"):
        preds = load_json(args.preds_file)
    else:
        raise ValueError(f"can not read model-generated summaries in {args.preds_file}")

    # read dialogue contents (source documents)
    if args.dataset_name == "dialogsum":
        data_instances = load_jsonl(args.data_file)
        assert len(preds) == len(data_instances)
    elif args.dataset_name == "samsum":
        data_instances = load_json(args.data_file)
        assert len(preds) == len(data_instances)
    else:
        raise ValueError(f"dataset not supported: {args.dataset_name}")

    dialogues = [x[args.dlg_column] for x in data_instances]
    references = [x[args.reference_column] for x in data_instances]
    assert len(preds) == len(dialogues) == len(references)

    src_list = dialogues
    ref_list = references
    output_list = preds

    data = convert_to_json(output_list=output_list,
                           src_list=src_list, ref_list=ref_list)
    # Initialize evaluator for a specific task
    evaluator = get_evaluator(task)
    # Get multi-dimensional evaluation scores
    eval_scores = evaluator.evaluate(data, print_result=True)

    pickle_dump(eval_scores, args.output_file)