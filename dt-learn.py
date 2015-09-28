# -*- coding: utf-8 -*-
__author__ = 'Danny'

import sys
import pandas as pd
import math

CLASS_VALUE = {"tested_negative\n":0,"tested_positive\n":1}


##################################################
# parse ARFF
##################################################
def parseARFF(fname):
    head = []
    accepting_data = False
    data =[]
    with open(fname) as arffStr:
        for line in arffStr:
            if accepting_data:
                strarray = line.split(",")
                data.append([float(val) for val in strarray[0:-1]]+[CLASS_VALUE[strarray[-1]]])

            elif line.find("@attribute")>-1:
                strlist = line.split("'")
                head.append(strlist[1])

            if line.find("@data")>-1:
                accepting_data = True

    return pd.DataFrame(data=data,columns=head)


##################################################
# tree node class that will make up the tree
##################################################
class TreeNode():
    def __init__(self, classification=None, parent=None):
#        self.is_leaf = True
        self.classification = classification
        self.attr = None
        self.partition_values = []
        self.parent = parent
        self.children = []
        self.entropy = 0
        self.majority = None


##################################################
# MakeSubtree(set of training instances D)
##################################################
def Maketree(examples, features, m):


    #Create a Root node for the tree
    root = TreeNode()
    num_class1 = float(examples[examples['class']==1].shape[0])
    total_num = float(examples.shape[0])

    #If the number of instances that reach a leaf node is 0,
    # the leaf should predict the plurality class of the parent node.
    if total_num==0:
        root.classification = root.parent.majority
        return root

    if total_num-num_class1==num_class1:
        root.majority = examples.ix[0]['class']
    elif total_num-num_class1>num_class1:
        root.majority = 0
    else:
        root.majority = 1

    #(i) all of the training instances reaching the node belong to the same class
    if num_class1==0 or total_num==num_class1:
        root.classification = root.majority
        return root

    #(ii) there are fewer than m training instances reaching the node
    if total_num<m:
        root.classification = root.majority
        return root
    else:
    #   otherwise make an internal node N
    #   FindBestSplit S
        root.entropy = -num_class1/total_num*math.log(num_class1/total_num,2)-(total_num-num_class1)/total_num*math.log((total_num-num_class1)/total_num,2)
        max_gain =0
        for feature in features:
            C = DetermineCandidateSplits(examples,feature)
            if len(C)==0:
                continue

            C.insert(0,float('-inf'))
            C.append(float('inf'))

            h = 0
            for i in range(len(C)-1):
                query = examples[examples[feature]<=C[i+1]][examples[feature]>C[i]]
                query0 = query[examples['class']==0]
                lenquery = float(query.shape[0])
                lenquery0 = float(query0.shape[0])
                lenquery1 = lenquery - lenquery0
                if lenquery0!=0:
                    h += -lenquery0/total_num*math.log(lenquery0/lenquery,2)
                if lenquery1!=0:
                    h += -lenquery1/total_num*math.log(lenquery1/lenquery,2)

            if (root.entropy-h)>max_gain:
                S = C
                max_gain = root.entropy-h
                best_feature = feature

        # (iii) no feature has positive information gain
        # (iv) there are no more remaining candidate splits at the node
        if max_gain==0:
            return root
        else:
            root.attr = best_feature
            root.partition_values = S

            for i in range(len(C)-1):
                child = Maketree(examples[examples[feature]<=C[i+1]][examples[feature]>C[i]].reset_index(),features,m)
                root.children.append(child)
                child.parent = root

    return root
    #       for each outcome k of S
    #           Dk = subset of instances that have outcome k!
    #           kth child of N = MakeSubtree(Dk)
    #  return subtree rooted at N


def DetermineCandidateSplits(examples,feature):
    C = [] #initialize set of candidate splits for feature Xi

    #partition instances in D into sets s1 ... sV where the instances in each set have the same value for Xi
    sorted_data = examples.sort([feature])
    row_iterator = sorted_data.iterrows()
    i,last = row_iterator.next()
    partition = [[i]]
    pt_ct = 0       # == len(partition) -1
    for i,row in row_iterator:
        if row[feature]==last[feature]:
            partition[pt_ct].append(i)
        else:
            partition.append([i])
            pt_ct += 1
            last = row

    #let vj denote the value of Xi for set sj
    #sort the sets in S using vj as the key for each sj
    #for each pair of adjacent sets sj, sj+1 in sorted S
    for j in range(pt_ct):
        for s in range(len(partition[j])):
            for t in range(len(partition[j+1])):
                #if sj and sj+1 contain a pair of instances with different class labels!
                if examples.ix[partition[j][s]]['class']!=examples.ix[partition[j+1][t]]['class']:
                   #add candidate split Xi ≤ (vj + vj+1)/2 to C!
                    C.append((examples.ix[partition[j][0]][feature]+
                                        examples.ix[partition[j+1][0]][feature])/2)
                    break
            else:
                continue
            break

    return C


def main():
    #fname = sys.argv[1]
    fname = "diabetes_train.arff"
    examples = parseARFF(fname)
    m = 2
    treeroot = Maketree(examples,examples.columns[:-1],m)


if __name__ == "__main__":
	main()