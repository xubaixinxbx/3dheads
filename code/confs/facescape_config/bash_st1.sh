pwd
baseid=(-1)
sp_view=10
for item in ${baseid[@]}
do
echo $sp_view
echo $item
python training/exp_runner.py \
--nepoch 3000 \
--conf ./confs/fs_real/st1.conf --scan_id ${item} --view_num ${sp_view} \
--gpu 3 
done