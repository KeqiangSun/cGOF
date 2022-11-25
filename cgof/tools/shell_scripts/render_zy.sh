# CUDA_VISIBLE_DEVICES=1 python render_multiz_to_one.py outputs/$1/generator.pth --range 0 100 1 --curriculum $1  --save_depth&
python render.py --device $1 --fn $2