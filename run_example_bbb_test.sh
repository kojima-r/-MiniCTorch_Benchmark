mkdir -p log

export  MKL_NUM_THREADS=1

b=16
postfix=b${b}
echo $postfix
python benchmark/example_bbb.py --num_layer 2 --batch_size ${b} --output_dir example_bbb_b${b}

#python benchmark/example_bbb.py --num_layer 2 --batch_size ${b} --output_dir example_bbb_b${b} --torch_exec >> ${log}

cd example_bbb_b${b}
make -j4

./bbb_train 

cd ..


