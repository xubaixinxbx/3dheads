pwd
sp_view=15
baseid=(371)
for item in ${baseid[@]}
do
echo $sp_view
echo $item
python training/exp_runner.py --is_continue \
--checkpoint 5000 --nepoch 20000 \
--conf ./confs/face_st2.conf --scan_id ${item} --view_num ${sp_view} \
--gpu 3
done