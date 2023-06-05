mkdir ../../data/logs

domain_name="humanoid"
task_name="realworld_walk"
level="easy"

gpu=0
for scheduler in "constant" "random_walk" "drift_pos" "drift_neg" "cyclic_pos" "cyclic_neg" "uniform" "saw_wave"
do
    CUDA_VISIBLE_DEVICES=${gpu} nohup python -u run_dmpo_acme.py --domain_name=${domain_name} --task_name=${task_name} --scheduler=${scheduler} --level=${level} > ../../data/logs/dmpo_acme_${domain_name}_${task_name}_${scheduler}_${level}.log &
    gpu=$((gpu+1))
done 

