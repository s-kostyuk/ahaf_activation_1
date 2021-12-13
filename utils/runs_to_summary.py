#!/usr/bin/env python3

import csv
import itertools
import os.path

from typing import Sequence, Dict, Tuple, Iterable, List, MutableSequence
from decimal import Decimal


def tuple_to_file_name(options: Sequence[str]) -> str:
    net = options[0]
    var = options[1]
    trainer = options[2]
    activation = options[3]
    dataset = options[4]

    net_str = "{}{}_ul".format(net, var) if var else net
    trainer_str = "_" + trainer if trainer else ""

    return "dynamics_{}_{}_{}_bs{}{}_ep{}.csv".format(
        net_str, dataset, activation, 64, trainer_str, 100
    )


def find_most_accurate(data: Iterable[Dict[str, float]]) -> Tuple[int, float]:
    max_acc = -1.0
    max_pos = -1
    pos = 0

    for el in data:
        pos += 1
        acc = Decimal(el["test_acc"]) * 100

        if acc > max_acc:
            max_acc = acc
            max_pos = pos

    return max_pos, max_acc


def gather_results(combinations: Iterable[Sequence[str]], runs_dir: str = "runs") -> List[MutableSequence]:
    results = []
    for options in combinations:
        file_name = tuple_to_file_name(options)
        file_path = "{}/{}".format(runs_dir, file_name)

        print("------------------------------------------------")

        if os.path.exists(file_path):
            print("Discovered data for: {}".format(options))
        else:
            print("No such file or directory for: {}".format(options))
            print("Skipped this combination of options")
            results.append(
                [*options, -1, -1.0]
            )
            continue

        # fields = ("epoch", "train_loss_mean", "train_loss_var", "test_acc", "lr")
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            pos, acc = find_most_accurate(reader)
            results.append(
                [*options, pos, acc]
            )

    return results


def results_to_table(results: List[Sequence], file_name: str):
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(
            (
                ("Network", "Activation", "DSPT", "F-MNIST", "", "CIFAR-10", ""),
                ("", "", "", "Accuracy, %", "Epoch", "Accuracy, %", "Epoch")
            )
        )

        right = []

        for result in results:
            net = result[0]
            var = result[1]
            trainer = result[2]
            act = result[3]
            dataset = result[4]
            ep = result[5]
            acc = result[6]

            net_str = "LeNet-5" if net == "LeNet" else net
            act_str = "{}-like {}".format(act, var.upper()) if var else act

            if var:
                trainer_str = "TRUE" if isinstance(trainer, str) and "dsu" in trainer else "FALSE"
            else:
                trainer_str = "N/A"

            right.extend(
                (acc, ep)
            )

            if len(right) == 4:
                writer.writerow(
                    (net_str, act_str, trainer_str, *right)
                )
                right = []

    print("Processed to CSV: {} experiments".format(len(results)))


def results_to_latex(results: List[Sequence], file_name: str):
    header = """
\t\\begin{table}[htbp]
\t\t\\label{table:tab1}
\t\t\\caption{Best test set accuracy, up to 100 epochs}
\t\t\\begin{tabular}{llllcccc}
\t\t\t\\toprule
\t\t\t& & & & \\multicolumn{2}{c}{Fashion-MNIST} & \\multicolumn{2}{c}{CIFAR-10} \\\\
\t\t\t\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}
\t\t\tNetwork & Activ. & Init. & Proc. & Acc. & Epoch & Acc. & Epoch \\\\
\t\t\t\\midrule"""

    line_template = "\n\t\t\t{} & {} & {} & {} & {}\\% & {} & {}\\% & {} \\\\"

    footer = """
\t\t\t\\bottomrule
\t\t\\end{tabular}
\t\\end{table}
"""

    with open(file_name, 'w') as f:
        f.write(header)

        right = []

        for result in results:
            net = result[0]
            var = result[1]
            trainer = result[2]
            act = result[3]
            dataset = result[4]
            ep = result[5]
            acc = result[6]
            best = False if len(result) < 8 else result[7]

            net_str = "LeNet-5" if net == "LeNet" else net
            act_str = var.upper() if var else act
            init_str = act if var else "N/A"
            acc_str_base = "{:.2f}".format(acc)
            acc_str = "\\textbf{{{}}}".format(acc_str_base) if best else acc_str_base

            if var and isinstance(trainer, str) and "dsu" in trainer:
                trainer_str = "DSPT"
            else:
                trainer_str = "Classic"

            right.extend(
                (acc_str, ep)
            )

            if len(right) == 4:
                line_str = line_template.format(net_str, act_str, init_str, trainer_str, *right)
                f.write(line_str)
                right = []

        f.write(footer)

    print("Processed to Latex: {} experiments".format(len(results)))


def results_add_best_flag(results: List[MutableSequence]):
    best_refs = {}
    curr_net = None
    curr_ds = None

    for r in results:
        net = r[0]
        ds = r[4]
        acc = r[6]
        options = (net, ds)

        if len(r) < 8:
            r.append(False)

        best_for_options = best_refs.get(options, None)
        if best_for_options is None:
            best_refs[options] = r
            continue

        best_acc = best_for_options[6]
        if acc > best_acc:
            best_refs[options] = r
            continue

    for el in best_refs.values():
        el[7] = True


def main():
    networks = ["LeNet", "KerasNet"]
    net_variants = ["", "Ahaf"]
    trainers = ["", "dsu4"]
    activations = ["ReLU", "SiL"]
    datasets = ["F-MNIST", "CIFAR10"]

    all_combinations = itertools.product(networks, net_variants, trainers, activations, datasets)
    all_combinations = filter(lambda x: x[1] or not x[2], all_combinations)

    results = gather_results(all_combinations, runs_dir="runs")
    results_add_best_flag(results)
    results_to_table(results, "runs/summary.csv")
    results_to_latex(results, "runs/summary.tex")


if __name__ == "__main__":
    main()
