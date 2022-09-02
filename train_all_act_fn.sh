if [[ $1 = "MNIST" ]]; then
    filename="mnist_main.py";
    epochs=20;
elif [[ $1 = "CIFAR10" ]]; then
    filename="main.py";
    epochs=50;
else
    echo "Usage ./${0} dataset";
    echo "Supported datasets: MNIST, CIFAR10"
    exit -1
fi

echo "File name is $filename";
echo "Epochs is $epochs";

for i in {1..8};
do
    echo "$i"
    python $filename --act_fn="$i" --max_epochs=$epochs
done
