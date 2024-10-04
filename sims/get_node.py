import re
import dendropy as dp

h1_flat = '../data/trees_alignments/flatfishes/manual_change_astral-20-noSSR_names_updates.tree'
h2_flat = "../data/trees_alignments/flatfishes/ml_tree_inter_taxa_properly_rooted.tree"

with open(h1_flat, "r") as f:
    h1 = dp.Tree.get(data = f.read(), schema = 'newick', preserve_underscores = True)

with open(h2_flat, "r") as f:
    h2 = dp.Tree.get(data = f.read(), schema = 'newick', preserve_underscores = True)


h1_taxa = set( [i.taxon.label for i in h1.leaf_node_iter()] )

pattern = "(Psettodidae|Achiridae)"
target_taxa = [i for i in h1_taxa if re.findall(pattern, i)]

# used the label to locate the node
h1.mrca(taxon_labels = target_taxa)



