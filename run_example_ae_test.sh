mkdir -p log

export  MKL_NUM_THREADS=1

z=4
postfix=z${z}
echo $postfix
python benchmark/example_ae.py --dim_z ${z} --batch_size 32 --output_dir example_ae_z${z}
#python benchmark/example_ae.py --dim_z ${z} --batch_size 32 --output_dir example_ae_z${z} --torch_exec

cd example_ae_z${z}
make -j4

./ae_train 
cd ..



