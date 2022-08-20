import os
import json
import collections

def get_loss_acc(line):
    splits = line.split(" | ")
    loss = float(splits[-2].split()[1])
    acc = float(splits[-1].split()[1][:-1])
    return loss, acc

def process_text(text):
    train_loss_acc = []
    val_loss_acc = []
    model_name = ''

    for line in text:
        if 'Model:' in line:
            model_name = line.rpartition(' ')[2][:-1]
        elif '| Train' in line:
            train_loss_acc.append(get_loss_acc(line))
        elif '| Val' in line:
            val_loss_acc.append(get_loss_acc(line))
    
    if not model_name:
        raise ValueError("Model name not found.")
    
    if not train_loss_acc or not val_loss_acc:
        raise ValueError("Invalid log file.")

    return {model_name: [train_loss_acc, val_loss_acc]}

def write_json(model_logs, data_filename):
    keys = ['model_name', 'train_loss', 'train_acc', 'val_loss', 'val_acc']
    data = []

    for (model, log) in model_logs.items():
        train_loss_acc, val_loss_acc = log
        print(model)
        tloss, tacc = map(list, zip(*train_loss_acc))
        vloss, vacc = map(list, zip(*val_loss_acc))
        values = [model, tloss, tacc, vloss, tacc]
        new_data = {k: v for (k, v) in zip(keys, values)}

        # Bad practice
        if os.path.isfile(data_filename):    
            with open(data_filename) as f:
                data = json.load(f)

        data.append(new_data)

        with open(data_filename, 'w') as f:
            json.dump(data, f, indent=2)

if __name__ == "__main__":
    """
    d = './logs/MNIST/'
    logs = os.listdir(d)

    # filename = "./logs/MNIST/BasicConvModel-GELU-18-08-2022-12:44:54-epochs-20.log"
    for filename in logs:
        with open(os.path.join(d, filename), "r") as f:
            text = f.readlines()

        data = process_text(text)
        write_json(data, "mnist_log.json")
    """

    # CIFAR10
    d = './logs/'
    logs = os.listdir(d)
    data_filename = "cifar10_log.json"

    # CIFAR10 has multilpe logs due to multiple runs 
    # Pick the latest log
    act_fn_map = collections.defaultdict(lambda: '')

    for filename in logs:
        if len(splits := filename.split('-')) < 2:
            continue
        else:
            key = splits[1]
        
        if act_fn_map[key] < (path := os.path.join(d, filename)):
            act_fn_map[key] = path

    print(act_fn_map)

    if os.path.isfile(data_filename):    
        os.remove(data_filename)

    for (_, filename) in act_fn_map.items():
        with open(filename) as f:
            text = f.readlines()

        data = process_text(text)
        write_json(data, data_filename)
    
    with open(data_filename) as f:
        data = json.load(f)

    data = {"data": data}

    with open(data_filename, 'w') as f:
        json.dump(data, f, indent=2)