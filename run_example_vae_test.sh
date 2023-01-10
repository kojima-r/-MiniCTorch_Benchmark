mkdir -p log

export  MKL_NUM_THREADS=1

z=4
postfix=z${z}
echo $postfix
python benchmark/example_vae.py --dim_z ${z} --batch_size 32 --output_dir example_vae_z${z}
trial=2
echo "Trial: ${trial}"
python benchmark/example_vae.py --dim_z ${z} --batch_size 32 --output_dir example_vae_z${z} --torch_exec

cd example_vae_z${z}
make -j4

echo "Trial: ${trial}" 
./vae_train
cd ..



