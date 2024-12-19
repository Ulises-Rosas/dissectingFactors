import argparse
import random
import dendropy as dp
from collections import deque

# tree_file = "/Users/ulises/Desktop/ABL/software/dissectingFactors/sims/prota_sims/h2_constrained.treefile"


def get_random_subset(tree_file, prop, prob, seed, outname):
    # twohypos="/Users/ulisesrosas/Desktop/dissectingFactors/sims/prota_sims/prota_myboth_hypos_v2.txt"

    random.seed(seed)

    all_strings=deque()
    sample_string = ''
    with open(tree_file, "r") as f:
        for line in f.readlines():
            tmp_line = line.strip()
            

            if not sample_string:
                sample_string = tmp_line
            
            all_strings.append(tmp_line)

    tmp_tree = dp.Tree.get_from_string(sample_string, 'newick', preserve_underscores=True)
    all_names = [i.taxon.label for i in tmp_tree.leaf_node_iter()] 


    xi = random.random()
    if xi < prob:
        all_names = random.sample(all_names, k=int(len(all_names)*prop))


    new_strings = [0]*len(all_strings)
    k=0
    while all_strings:
        s = all_strings.popleft()

        tmp_tree = dp.Tree.get_from_string(s, 'newick', preserve_underscores=True)
        tmp_tree.retain_taxa_with_labels(set(all_names))

        new_strings[k] = tmp_tree.as_string("newick")
        k+=1



    with open(outname, "w") as f:
        for s in new_strings:
            f.write(s.strip().replace("'", ""))
            f.write("\n")
        # f.write(tmp_tree.as_string("newick"))
    

def args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('tree_file', type=str, help='Path to the tree file')
    parser.add_argument('--prop', type=float, help='Proportion of the original tree to keep')
    parser.add_argument('--prob', type=float, help='Probability of keeping the tree')
    parser.add_argument('--seed', type=int, help='Seed for the random number generator')
    parser.add_argument('--outname', type=str, help='outname for the new tree file')
    return parser.parse_args()

def main():
    args_ = args()
    get_random_subset(args_.tree_file, args_.prop, args_.prob, args_.seed, args_.outname)


if __name__ == "__main__":
    main()


# def main():
# seed = 12038
# prop = 0.8
# prob = 0.5


