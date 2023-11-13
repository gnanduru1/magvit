singularity	exec --env PYTHONPATH=/home/bae9wk/.local/lib/python3.10/site-packages \
    --nv magvit_env.sif \
    python videogvt/main.py \
    --config=/scratch/bae9wk/magvit/videogvt/configs/vqgan3d_ssv2_config.py \
    --workdir=workdir
