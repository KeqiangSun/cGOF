curriculum=$1
echo("CUDA_VISIBLE_DEVICES=1 python render_multiz_to_one.py outputs/${curriculum}/generator.pth --range 0 100 1 --curriculum ${curriculum}  --save_depth&")
