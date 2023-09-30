pwd
sp_view=15
baseid=(-1)
for item in ${baseid[@]}
do
echo $sp_view
echo $item
python training/exp_runner.py \
--nepoch 2000 \
--conf ./confs/face_st1.conf --scan_id ${item} --view_num ${sp_view} \
--gpu 1
done