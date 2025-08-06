if [[ -z "$1" ]]; then
    echo "Usage: $0 <task-name>"
    return 1
fi

basefolder=$(dirname $0)

rm $basefolder/smoothed_csv/*

echo "Copying $(realpath $basefolder/../logs/csv/worker/$1/*) into $(realpath $basefolder/original_csv/$1)/"
mkdir -p $basefolder/original_csv/$1
cp $basefolder/../logs/csv/worker/$1/* $basefolder/original_csv/$1
python3 $basefolder/smooth_original_csv.py $basefolder/original_csv/$1

if [ "$(ls "$basefolder/../logs/csv/worker/$1" 2>/dev/null | wc -l)" -ge 2 ]; then
    python3 create_comp_graph.py smoothed_csv/*.csv
else 
    python3 create_graph.py smoothed_csv/$(basename $basefolder/../logs/csv/worker/$1/*)
fi

