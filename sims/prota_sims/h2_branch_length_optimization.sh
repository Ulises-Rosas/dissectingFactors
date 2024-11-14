aln='/Users/ulises/Desktop/GOL/data/alldatasets/nt_aln/internally_trimmed/malns_36_mseqs_27/round2/no_lavaretus/McL/mysupermatrix.txt'
partitions='/Users/ulises/Desktop/GOL/data/alldatasets/nt_aln/internally_trimmed/malns_36_mseqs_27/round2/no_lavaretus/McL/mypartitions.txt'
twohypos='/Users/ulises/Desktop/ABL/software/dissectingFactors/sims/prota_sims/prota_myboth_hypos.txt'

sed -n 2p $twohypos > tree_h2.txt


# error at 9558
iqtree2 -s $aln -p $partitions -g tree_h2.txt --prefix h2_constrained -T 15


