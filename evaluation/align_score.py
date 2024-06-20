import json
from argparse import ArgumentParser

from alignscore import AlignScore


def load_jsonl(filename):
    lines = []
    with open(filename, "r") as file:
        for line in file:
            lines.append(json.loads(line))
    return lines


def load_json(filename):
    with open(filename, "r") as file:
        return json.load(file)


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument("--preds_file", type=str, help="model prediction (generated summaries) file.")
    argparser.add_argument("--data_file", type=str, help="data file for prediction, containing original dialogues.")
    argparser.add_argument("--dlg_column", type=str, default="dialogue")
    argparser.add_argument("--dataset_name", type=str, choices=["samsum", "dialogsum"])
    argparser.add_argument("--alignscore_model_path", type=str, help="path of alignscore model checkpoint.")
    argparser.add_argument("--output_file", type=str, help="output file to save scores.")
    argparser.add_argument("--device", type=str, default="cpu")
    argparser.add_argument("--eval_batch_size", type=int, default=8)

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
        dialogues = [x[args.dlg_column] for x in data_instances]
        assert len(preds) == len(dialogues)
    elif args.dataset_name == "samsum":
        data_instances = load_json(args.data_file)
        dialogues = [x[args.dlg_column] for x in data_instances]
        assert len(preds) == len(dialogues)
    else:
        raise ValueError(f"dataset not supported: {args.dataset_name}")

    scorer = AlignScore(
        model='roberta-base',
        batch_size=args.eval_batch_size,
        device=args.device,
        ckpt_path=args.alignscore_model_path,
        evaluation_mode='nli_sp')
    score = scorer.score(
        contexts=dialogues,
        claims=preds
    )

    with open(args.output_file, "w") as file:
        json.dump(score, file, indent=4)