for i in {1..5};
do
    echo "$i"
    python main.py --act_fn="$i" --max_epochs=50
done