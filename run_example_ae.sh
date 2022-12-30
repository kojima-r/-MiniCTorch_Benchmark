mkdir -p log

export  MKL_NUM_THREADS=1


for z in 2 4 6 8 16 32 64 128 256
do
postfix=z${z}
echo $postfix
log=log/example_ae_z${z}.txt
python benchmark/example_ae.py --dim_z ${z} --batch_size 32 --output_dir example_ae_z${z} > ${log}

for trial in `seq 2 5`
do
echo "Trial: ${trial}" >> ${log}
python benchmark/example_ae.py --dim_z ${z} --batch_size 32 --output_dir example_ae_z${z} --torch_exec >> ${log}
done

cd example_ae_z${z}
make -j4

for trial in `seq 1 5`
do
echo "Trial: ${trial}" >> ../${log}
./ae_train >> ../${log}
done

cd ..
done



