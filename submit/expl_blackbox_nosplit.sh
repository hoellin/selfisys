#!/bin/bash
#PBS -S /bin/sh
#PBS -N expl_blackbox
#PBS -j oe
#PBS -o /home/hoellinger/selfi_sys_public/submit/logs/expl_blackbox.log
#PBS -l nodes=1:ppn=32,walltime=2:00:00

module ()
{
    eval `/softs/environment-modules/latest/bin/modulecmd bash --color $* `
}
module use /usr/share/Modules/modulefiles
module use ~/.modulefiles/

module load "/home/hoellinger/.modulefiles/anaconda/python3.10_2023.03"
source activate coca
export LD_LIBRARY_PATH=/home/hoellinger/.local/apps/gsl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/hoellinger/.local/apps/fftw/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/hoellinger/.local/apps/hdf5/lib:$LD_LIBRARY_PATH
export HDF5_DISABLE_VERSION_CHECK=1

export OMP_NUM_THREADS=32
python3 "/home/hoellinger/selfi_example_sys/src/expl/expl_blackbox_nosplit_parser.py" \
    --wd_ext march19th/ \
    --name run1 \
    --size 256 \
    --Np0 256 \
    --Npm0 256 \
    --L 500 \
    --S 64 \
    --Pinit 50 \
    --Nnorm 1 \
    --total_steps 20 \
    --sim_params "custom19COLA20RSD" \
    --aa 0.05 0.84 0.88 0.95 \
    --OUTDIR "/data101/hoellinger/selfi_sys_public/" \
    --radial_selection "multiple_lognormal" \
    --selection_params 0.031 0.034 0.036 0.045 0.129 0.19 1 1 1 \
    --survey_mask_path "/home/hoellinger/selfi_example_sys/data/extinction_map/raw_mask_Mar16_N256.npy" \
    --lin_bias 1.3 1.4 1.5 \
    --noise 0.2 \
    --force False \
    --verbosity 2
