python slurm_submitter.py --n 30 --window 30\
                          --params indel_params_0.02_0.06.txt\
                          --iqtree ../../../../iqtree-2.3.6-Linux-intel/bin/iqtree2\
                          --random_len random_len.py\
                          --random_prunning random_prunning.py\
                          --twohypos prota_myboth_hypos_v2.txt 


cp sim_*/*_au.iqtree sim_progress
cp sim_*/*_gt.treefile sim_progress
cp sim_*/*.fa sim_progress

# put into python list a and check the number of even numbers
ls *_au.iqtree  | sed -Ee "s/indel_(.+)_au.iqtree/\1/g" | sort -nr | sed -Ee "s/(.+)/\1,/g"
len([i for i in a if i % 2 == 0])


ggpy features -A .fa -T _gt.treefile -c -n 18
python ../sim_files/process_au_files.py *_au.iqtree -o au_support_700.txt


python ../../src/FI_DNN_v3_sims.py features_stats.tsv au_support_700.txt\
                  -p test --n_epochs 10 --ncpus 10 --max_trials 2 --tu_epochs 10\
                  --iterations 2 --prefix test_v2

# time python ../../src/FI_DNN_v3_sims.py features_stats.tsv au_support_700.txt \
#                       -p test --n_epochs 2000 --ncpus 10 --max_trials 10 --tu_epochs 2000\
#                       --iterations 20 --prefix test_v1
