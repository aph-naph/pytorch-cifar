import csv

from collections import defaultdict

def get_loss_acc(line):
    left, _, right = line.partition("|")
    loss = float(left.split()[-1])
    acc = float(right.split()[1][:-1])
    return loss, acc

def process_text(text, start=None, stop=None):
    model_results = defaultdict(list)
    # train_loss_acc = []
    # val_loss_acc = []

    model_name = ''
    started = False if start else True

    for line in text:
        if start and start in line:
            started = True

        if not started:
            continue

        if stop and stop in line:
            print("Encountered", stop)
            print("Stopping.")
            break

        if 'Model:' in line:
            model_name = line.split()[1]

        if 'Train -' in line:
            train = get_loss_acc(line)
            model_results[model_name].append(train)
        
        if 'Val' in line:
            val = get_loss_acc(line)
            model_results[model_name].append(val)
        
        if 'Test' in line:
            test = get_loss_acc(line)
            model_results[model_name].append(test)
            model_name = ''

    return model_results

def write_csv(model_results, filename):
    f = open(filename, 'w', newline='')
    csvwriter = csv.writer(f, delimiter=',')
    
    csvwriter.writerow(['Model Name', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc'])

    for (model, results) in model_results.items():
        results = [j for i in results for j in i]
        csvwriter.writerow([model] + results)
    
    f.close()


if __name__ == "__main__":
    filename = "research_notes.txt"

    with open(filename, "r") as f:
        text = f.readlines()

    results = process_text(text, stop='MNIST')
    write_csv(results, "cifar10_results.csv")
    # results = process_text(text, start='MNIST')
    # write_csv(results, "mnist_results.csv")
    print(results)