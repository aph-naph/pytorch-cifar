import os
import json
import collections

def get_loss_acc(line):
    splits = line.split(" | ")
    loss = float(splits[-2].split()[1])
    acc = float(splits[-1].split()[1][:-1])
    return loss, acc

def process_texts(texts):
    log_data = {}
    for text in texts:
        train_loss_acc = []
        val_loss_acc = []
        test_loss_acc = None
        model_name = ''

        for line in text:
            if 'Model:' in line:
                model_name = line.rpartition(' ')[2][:-1]
            elif '| Train' in line:
                train_loss_acc.append(get_loss_acc(line))
            elif '| Val' in line:
                val_loss_acc.append(get_loss_acc(line))
            elif '| Test' in line:
                test_loss_acc = get_loss_acc(line)
        
        if not model_name:
            raise ValueError("Model name not found.")
        
        if not train_loss_acc or not val_loss_acc:
            raise ValueError("Invalid log file.")

        log_data.update({model_name: [train_loss_acc, val_loss_acc, test_loss_acc]})

    return log_data

def get_logs(logs_dir):
    # There maybe multiple logs due to multiple runs 
    # Pick the latest log
    act_fn_map = collections.defaultdict(lambda: '')

    for filename in os.listdir(logs_dir):
        if len(splits := filename.split('-')) > 2:
            key = splits[1] # Key is name of the activation function        
            if act_fn_map[key] < (path := os.path.join(d, filename)):
                act_fn_map[key] = path

    return act_fn_map.values()

def process_log_data(logs_data):
    keys = ['model_name', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc']
    data = []
    for (model_name, log) in logs_data.items():
        train_loss_acc, val_loss_acc, test_loss_acc = log
        print(model_name)
        trloss, tracc = map(list, zip(*train_loss_acc))
        vloss, vacc = map(list, zip(*val_loss_acc))
        tloss, tacc = test_loss_acc
        values = [model_name, trloss, tracc, vloss, tracc, tloss, tacc]
        new_data = {k: v for (k, v) in zip(keys, values)}
        data.append(new_data)

    return data

if __name__ == "__main__":

    # MNIST

    d = './logs/MNIST'
    data_filename = "mnist_log_rmsprop.json"
    log_files = get_logs(d)
    log_texts = []

    for log_file in log_files:
        with open(log_file, "r") as f:
            log_texts.append(f.readlines())

    logs_data = process_texts(log_texts)
    data = process_log_data(logs_data)

    with open(data_filename, 'w') as f:
        json.dump({"data": data}, f, indent=2)

    # CIFAR10

    d = './logs/CIFAR10'
    logs = os.listdir(d)
    data_filename = "cifar10_log_rmsprop.json"
    log_files = get_logs(d)
    log_texts = []

    for log_file in log_files:
        with open(log_file, "r") as f:
            log_texts.append(f.readlines())
        
    logs_data = process_texts(log_texts)
    data = process_log_data(logs_data)

    with open(data_filename, 'w') as f:
        json.dump({"data": data}, f, indent=2)