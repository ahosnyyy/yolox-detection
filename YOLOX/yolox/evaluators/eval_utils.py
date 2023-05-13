import os
import json

def dump_json(lst, out_dir, file_name):
    obj = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
    full_json_path = os.path.join(out_dir, file_name)
    with open(full_json_path, "w") as f:
        json.dump(obj, f, indent=4, separators=(",", ": "))


def coco_eval_json(values, out_dir):
    obj = {'AP@0.50:0.95': 0, 'AP@0.50': 0, 'AP@0.75': 0, 'AR@0.50:0.95': 0}
    obj['AP@0.50:0.95'] = values[0]
    obj['AP@0.50'] = values[1]
    obj['AP@0.75'] = values[2]
    obj['AR@0.50:0.95'] = values[8]

    # Save the result to a JSON file
    full_json_path = os.path.join(out_dir, "val_stats.json")
    with open(full_json_path, 'w') as f:
        json.dump(obj, f, indent=4, separators=(",", ": "))


def time_json(a_infer_time, a_nms_time, out_dir):
    pairs = [
        ("forward", a_infer_time),
        ("NMS", a_nms_time),
        ("inference", (a_infer_time + a_nms_time))
        ]

    # create a dictionary from the pairs
    data = dict(pairs)

    # save the dictionary to a JSON file
    full_json_path = os.path.join(out_dir, "val_time.json")
    with open(full_json_path, 'w') as f:
        json.dump(data, f, indent=4, separators=(",", ": "))
