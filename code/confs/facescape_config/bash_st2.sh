pwd
baseid=(340 344 212 122 20 19)
sp_view=3
for item in ${baseid[@]}
do
echo $sp_view
echo $item
python training/exp_runner.py --is_continue --checkpoint 1000 \
--nepoch 3000 \
--conf ./confs/fs_real/st2.conf --scan_id ${item} --view_num ${sp_view} \
--gpu 1
done