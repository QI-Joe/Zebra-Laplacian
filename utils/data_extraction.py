import re
import json
import os
import fnmatch
import numpy as np

def parse_test_metrics_from_text(text: str):
    """
    Parse test metrics from the given text.
    Returns a dictionary:
      {
        snapshot_index: {
          metric_name: { "max": float, "avg": float },
          ...
        },
        ...
      }
    """
    # Regex to find snapshot headers
    snapshot_pattern = re.compile(r"At snapshot (\d+), we get:")
    # Regex to find lines with test metrics, e.g.
    # test val_acc: max-0.5072815533980582 | | avg-0.47923408845738946
    test_metric_pattern = re.compile(
        r"test\s+([a-zA-Z0-9_]+): max-([0-9.eE+-]+) \| \| avg-([0-9.eE+-]+)"
    )

    lines = text.splitlines()
    data = {}
    current_snapshot = None

    for line in lines:
        # Check for snapshot header
        snap_match = snapshot_pattern.match(line.strip())
        if snap_match:
            current_snapshot = int(snap_match.group(1))
            data[current_snapshot] = {}
            continue

        # Within a snapshot, parse test metrics only
        if current_snapshot is not None:
            metric_match = test_metric_pattern.match(line.strip())
            if metric_match:
                metric_name = metric_match.group(1)
                max_val = float(metric_match.group(2))
                avg_val = float(metric_match.group(3))
                data[current_snapshot][metric_name] = {
                    "max": max_val,
                    "avg": avg_val,
                }

    return data

def main(dataset: str, date, task: str):
    standard = rf"log/{dataset}/{date}/{task}"
    os.makedirs(os.path.join(standard, "jsonVersion"), exist_ok=True)
    filelist = list()
    for filename in os.listdir(standard):
        if fnmatch.fnmatch(filename, f"*{task}*.txt"):
            file_path = os.path.join(standard, filename)
            filelist.append(file_path)
    symbol=0
    val_acc_list = list()
    val_f1_list = list()
    for filename in filelist:
        oaa = dict()
        with open(filename, "r", encoding="utf-8") as f:
            text = f.read()

        test_data = parse_test_metrics_from_text(text)
        snapshot_keys = test_data.keys()
        for dkey in snapshot_keys:
            for dmetric_key, dvalue in test_data[dkey].items():
                if dmetric_key not in oaa.keys():
                    oaa[dmetric_key] = {"max": [dvalue["max"]], "avg": [dvalue["avg"]]}
                else: 
                    oaa[dmetric_key]["max"].append(dvalue["max"])
                    oaa[dmetric_key]["avg"].append(dvalue["avg"])
        task_ratio = filename.split("_")[-5]
        output_filepath = os.path.join(standard, "jsonVersion", f"{symbol}_{task}_{task_ratio}_test_metrics.json")
        symbol += 1
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(oaa, f, indent=4)

        print(f"Extracted test metrics saved to {output_filepath}")
        val_acc_list.append(oaa["val_acc"])
        val_f1_list.append(oaa["val_f1"])
    
    length = len(val_acc_list)
    for i in range(length):
        test_acc_max, test_acc_max_std = np.mean(val_acc_list[i]["max"]), np.std(val_acc_list[i]["max"])
        test_acc_avg, test_acc_avg_std = np.mean(val_acc_list[i]["avg"]), np.std(val_acc_list[i]["avg"])
        test_f1_max, test_f1_max_std = np.mean(val_f1_list[i]["max"]), np.std(val_f1_list[i]["max"])
        test_f1_avg, test_f1_avg_std = np.mean(val_f1_list[i]["avg"]), np.std(val_f1_list[i]["avg"])
        print(f"test accuracy max: {test_acc_max*100:.2f} +- {test_acc_max_std*100:.2f}%, "
              f"test accuracy avg: {test_acc_avg*100:.2f} +- {test_acc_avg_std*100:.2f}%")
        print(f"test f1 max: {test_f1_max*100:.2f} +- {test_f1_max_std*100:.2f}%, "
              f"test f1 avg: {test_f1_avg*100:.2f} +- {test_f1_avg_std*100:.2f}%\n")
        
if __name__ == "__main__":
    task = "imbalance"
    date, dataset = "06", "dblp"
    
    main(dataset, date, task)