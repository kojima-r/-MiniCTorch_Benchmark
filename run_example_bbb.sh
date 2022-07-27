mkdir -p log

export  MKL_NUM_THREADS=1


for b in 2 4 6 8 16 32 64
do
postfix=b${b}
echo $postfix
log=log/example_bbb_b${b}.txt
python benchmark/example_bbb.py --num_layer 2 --batch_size ${b} --output_dir example_bbb_b${b} > ${log}

for trial in `seq 2 5`
do
echo "Trial: ${trial}" >> ${log}
python benchmark/example_bbb.py --num_layer 2 --batch_size ${b} --output_dir example_bbb_b${b} --torch_exec >> ${log}
done

cd example_bbb_b${b}
make -j4

for trial in `seq 1 5`
do
echo "Trial: ${trial}" >> ../${log}
./bbb_train >> ../${log}
done

cd ..
done



for l in 0 1 2 3 4
do
postfix=l${l}
echo $postfix
log=log/example_bbb_l${l}.txt
python benchmark/example_bbb.py --num_layer ${l} --batch_size 32 --output_dir example_bbb_l${l} > ${log}

for trial in `seq 2 5`
do
echo "Trial: ${trial}" >> ${log}
python benchmark/example_bbb.py --num_layer ${l} --batch_size 32 --output_dir example_bbb_l${l} --torch_exec >> ${log}
done

cd example_bbb_l${l}
make -j4

for trial in `seq 1 5`
do
echo "Trial: ${trial}" >> ../${log}
./bbb_train >> ../${log}
done

cd ..
done

