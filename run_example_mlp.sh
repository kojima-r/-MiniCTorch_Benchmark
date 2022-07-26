mkdir -p log
python benchmark/example_mlp.py > log/example_mlp.txt
cd example_mlp
make -j4
./mlp_train >> ../log/example_mlp.txt

