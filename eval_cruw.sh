for num in {20..30}
do
  python ./tools/dist_test.py configs/cruw_2022/jde/hr_l4_w1.py \
  --work_dir work_dirs/hr_l4_w1/20231106_072314 \
  --checkpoint work_dirs/hr_l4_w1/20231106_072314/epoch_${num}.pth \
  --testset 
done