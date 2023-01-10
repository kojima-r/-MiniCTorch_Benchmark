
export  MKL_NUM_THREADS=1

b=8
postfix=b${b}
echo $postfix
python benchmark/example_mlp.py --num_layer 2 --batch_size ${b} --output_dir example_mlp_b${b} 

#trial=2
#echo "Trial: ${trial}"
#python benchmark/example_mlp.py --num_layer 2 --batch_size ${b} --output_dir example_mlp_b${b} --torch_exec 

cd example_mlp_b${b}
make -j4

echo "Trial: ${trial}"
./mlp_train

cd ..

