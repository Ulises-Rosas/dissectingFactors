import os
import argparse


class myNode:

    def __init__(self, name = None, 
                 left = None, 
                 right = None, 
                 ancestor = None,
                 index = None, 
                 node_label = None,
                 branch_length = 0.0):
        self.name = name
        self.left = left
        self.right = right
        self.ancestor = ancestor
        self.index = index
        self.branch_length = branch_length
        self.label = node_label


def fas_to_dic(file):
    
    file_content = open(file, 'r').readlines()
    seqs_list   = []
    
    for i in file_content:
        line = i.strip()
        if line: # just if file starts empty
            seqs_list.append(line) 
    
    keys = [] 
    values = []    
    i = 0
    while(">" in seqs_list[i]):
        keys.append(seqs_list[i])
        i += 1 
        JustOneValue = []

        while((">" in seqs_list[i]) == False):
            JustOneValue.append(seqs_list[i]) 
            i += 1

            if(i == len(seqs_list)):
                i -= 1
                break

        values.append("".join(JustOneValue).upper().replace(" ", ""))
        
    return dict(zip(keys, values))


def writeT(p):

    # p = n3
    if p.left is None and p.right is None:
        return f"{p.name}:{p.branch_length}"
    
    else:
        ln = writeT(p.left)
        rn = writeT(p.right)
        nlabel = '' if not p.label else p.label

        return f"({ln},{rn}){nlabel}:{p.branch_length}"

def parseTree(all_nodes, root = -1):

    w = all_nodes[root]

    if isinstance(w.right, list):
        n1 = w.right[0]
        n2 = w.right[1]
        n3 = w.left

        return f"({writeT(n1)},{writeT(n2)},{writeT(n3)});"
    
    else:
        n1 = w.right
        n2 = w.left

        return f"({writeT(n1)},{writeT(n2)});"

def parseBinTree(all_nodes, root = -1):

    w = all_nodes[root]
    n1 = w.left
    n2 = w.right

    return f"({writeT(n1)},{writeT(n2)});"

def renaming_tips(all_keys, tree):
    for n,k in enumerate(all_keys):
        # n,k
        spps = k.replace(">", "")
        tree = tree.replace(f"'{n}'", spps)

    return tree

def parse_and_rename(all_nodes, all_keys):
    nwk_str = parseTree(all_nodes)
    for n,k in enumerate(all_keys):
        nwk_str = nwk_str.replace(f"'t{n}'", k.replace(">", ""))
    return nwk_str

def get_int_nodes(n, T):
    int_nodes = [0]*(n - 2)
    k = 0
    for i in range(2*n - 2):

        if not T[i].name:
            int_nodes[k] = T[i].index
            k += 1

    return int_nodes

def get_edges(nodes_indx, n, T, get_root = False):
    
    E = [[0,0]]*(n - 3)
    k = 0; root = 0
    for u in nodes_indx:

        if isinstance(T[u].right, list):
            root = u
            continue
        
        E[k] = [u, T[u].ancestor.index]
        k += 1

    if get_root:
        return E, root
    
    else:
        return E
    
def tokenize(tree):
    """
    split into tokens based on the characters '(', ')', ',', ':', ';'
    Huelsenbeck, programming for biologists, pp. 124-125

    parameters:
    -----------
    tree : str
        newick tree string

    returns:
    --------
    tokens : list
        list of tokens
    """
    # tree = mytree
    tokens = []
    ns_size = len(tree)
    i = 0
    while i < ns_size:
        c = tree[i]
        if c in '(),:;':
            tokens.append(c)
            i += 1

        else:
            j = i
            tempStr = ''
            while c not in '(),:;':
                tempStr += c
                j += 1
                c = tree[j]

            i = j
            tokens.append(tempStr)

    return tokens

def check_if_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    

def has_children(p):
    if p.left is None or p.right is None:
        return False
    else:
        return True

def build_up_nodes(tokens):

    nodes = []
    root = None
    p = None
    readingBranchLength = False
    readingLabel = False
    k = 0
    for i, tk in enumerate(tokens):

        k += len(tk)

        if tk == "(":
            n = myNode()
            nodes.append(n)
            if p is None:
                root = n
            
            else:
                n.ancestor = p
                if p.left is None:
                    p.left = n
                else:
                    if p.right is None:
                        p.right = n
                    else:
                        # unrooted tree
                        if p == root:
                            p.right = [p.right, n]

            p = n

        elif tk == "," or tk == ")":
            # move down a node
            p = p.ancestor

            if tk == ")" and not has_children(p):
                raise ValueError(f"Error: We expect two children per node. Check character {k}")
            
            # check if the next token is a number
            next_number = check_if_number(tokens[i+1])
            
            if tk == ")" and next_number:
                readingLabel = True
            
        elif tk == ":":
            readingBranchLength = True

        elif tk == ";":
            # end of tree
            if p != root:
                raise ValueError("Error: We expect to finish at the root node")

        else:
            if readingBranchLength:
                p.branch_length = float(tk)
                readingBranchLength = False

            elif readingLabel:
                p.label = float(tk)
                readingLabel = False
                
            else:
                n = myNode(name = tk.strip("''"))
                nodes.append(n)
                n.ancestor = p

                if p.left is None:
                    p.left = n
                else:
                    if p.right is None:
                        p.right = n
                    else:
                        # unrooted tree
                        if p == root:
                            p.right = [p.right, n]

                p = n

    return nodes, root

def parseNewickTree(mstr):
    # mstr = mytree
    tokens = tokenize(mstr)

    nodes, root = build_up_nodes(tokens)
    # print(nodes[0].right)

    indx = 0
    for n in nodes:
        if not n.name:
            n.index = indx
            indx += 1

    for n in nodes:
        if n.name:
            n.index = indx
            indx += 1

    return nodes, root

# mytree = "((t0:0.1,t1:0.2):0.3,t2:0.4);"
# nodes, root = parseNewickTree(mytree)
# print(parseTree(nodes, root = root.index))

def dfs_ur(n, path, paths):
    """
    for unrooted trees
    and vcv estimation (initially)
    """
    if n:
        path += [n.index]

        if n.name:
            paths.append(list(path)) 

        
        dfs_ur(n.left, path, paths)

        if n.ancestor:
            dfs_ur(n.right, path, paths)
        else:
            dfs_ur(n.right[0], path, paths)
            dfs_ur(n.right[1], path, paths)

        path.pop()

def dfs_r(n, path, paths):
    """
    for rooted trees
    and vcv estimation (initially)
    """
    if n:
        path += [n.index]

        if n.name:
            paths.append(list(path)) 

        dfs_r(n.left, path, paths)
        dfs_r(n.right, path, paths)

        path.pop()



def filter_files(files_path, n_alns, out_folder):

    for i in range(1,n_alns + 1):

        file1=os.path.join(files_path, f"indel_{i}_gt.treefile")

        if not os.path.exists(file1):
            print(f"File {file1} does not exist")
            continue

        with open(file1, 'r') as f:
            tree1 = f.readline().strip()

        nodes, root = parseNewickTree(tree1)
        n_leaves = len([n for n in nodes if n.name and root != n])

        if n_leaves != 119:
            continue

        print("Processing", i)
        os.system(f"cp {files_path}/indel_{i}_gt.treefile {files_path}/indel_{i}.fa {files_path}/indel_{i}_au.iqtree {out_folder}")


def myargs():
    parser = argparse.ArgumentParser(description='Process iqtree files.')
    parser.add_argument('files_path', type=str, help='path to the files')
    parser.add_argument('n_alns', type=int, help='number of alignments to process')
    parser.add_argument('out_folder', type=str, help='output folder')
    args = parser.parse_args()
    return args

def main():
    args = myargs()
    filter_files(args.files_path, args.n_alns, args.out_folder)

if __name__ == "__main__":
    main()
