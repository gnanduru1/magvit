singularity exec --env PYTHONPATH=/home/$(whoami)/.local/lib/python3.10/site-packages \
    --nv magvit_env.sif \
    python videogvt/main.py \
    --config=videogvt/configs/maskgvt_ssv2_config.py \
    --workdir=workdir
