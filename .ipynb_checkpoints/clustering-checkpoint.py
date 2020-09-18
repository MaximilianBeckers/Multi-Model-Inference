#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sklearn
from sklearn.cluster import AgglomerativeClustering, KMeans

#----------------------------------------------------------------
def kmeans(data, num_classes):
    
    clustering = KMeans(n_clusters=num_classes, random_state=0).fit(data);
    
    return clustering.cluster_centers_, clustering.labels_;

#----------------------------------------------------------------
def UPGMA(data, num_classes):
    
    clustering = AgglomerativeClustering(n_clusters=num_classes, linkage='average')
    clustering.fit(data);
    
    #get centers of clusters
    class_centers = np.zeros((num_classes, data.shape[1]));
    for tmp_class in range(num_classes):    
    
        class_centers[tmp_class, :] = get_class_center(data[clustering.labels_==tmp_class,:]);
    
    return class_centers, clustering.labels_;

#-----------------------------------------------------------------
def UPGMA_rmsd(data, rmsd_limit, min_cluster_size = 0):
    
    # Apply UPGMA clustering
    cluster=AgglomerativeClustering(n_clusters=2, compute_full_tree=True, linkage='ward')
    cluster.fit(data)
    
    num_structures = data.shape[0];
    
    # Assign Clusters
    NumMolList, mol_dictionary = CalcSizeAndAssignment(cluster.children_, num_structures);
   
    # filter out irrelevant clusters
    class_centers, class_labels = determine_relevant_clusters(data, cluster.children_, num_structures, mol_dictionary, rmsd_limit, min_cluster_size);
        
    return class_centers, class_labels;

#------------------------------------------------------------
def CalcSizeAndAssignment(children, Ndata):
    
    # Assigns molecules to the clusters of the UPGMA tree
    NumMolList=[]
    MolDict={}
    
    for i in range(len(children)):
        N=0
        mols_assigned=[]
        for j in range(len(children[i])):
            if children[i][j] < Ndata:
                N += 1
                mols_assigned.append(children[i][j])
            else:
                N += NumMolList[children[i][j]-Ndata]
                mols_assigned += MolDict[children[i][j]]
                
        NumMolList.append(N)
        MolDict[i+Ndata] = mols_assigned
        
    return NumMolList, MolDict

#----------------------------------------------------------------
def get_RMSD(structure_1, structure_2):
    
    rmsd = np.sqrt(np.mean(((structure_1 - structure_2)**2)))
    
    return rmsd;

#----------------------------------------------------------------
def get_class_center(coord_array):
    
    num_samples = coord_array.shape[0]
    min_dist = 1.0*10**10;
    center_index = 0;
    
    class_center = np.mean(coord_array, axis=0);
    
    #find structure closest to the mean of the cluster and return it
    for tmp_sample in range(num_samples):
               
        tmp_dist = np.sqrt(np.sum(np.square(coord_array[tmp_sample,:] - class_center)));

        if tmp_dist < min_dist:
            center_index = tmp_sample;
            min_dist = tmp_dist;

    return coord_array[center_index]

#----------------------------------------------------------------
def determine_relevant_clusters(data, children, Ndata, MolDict, rmsd_limit, MinClusterSize=0):
    
    # filter out irrelevant clusters and calculate MCS on selected clusters
    class_labels = np.ones((Ndata)) * (-1);
    currlayer=[Ndata*2 - 2];
    class_centers = np.zeros((200, data.shape[1]))
    class_counter = 0;
    
    while len(currlayer)>0:
        childlayer=[]
        
        for c in currlayer:
            if c>=Ndata: #i.e. a non-leaf node
    
                print("new node")
                tmp_class_center = get_class_center(data[MolDict[c],:])
                    
                if (children[c - Ndata][0] < Ndata):
                    print("single sample cluster 1")
                    mol_ind_1 = children[c - Ndata][0];
                    center_cluster_1 = data[mol_ind_1, :];
                else:
                    mol_ind_1 = MolDict[children[c - Ndata][0]];
                    center_cluster_1 = get_class_center(data[mol_ind_1, :]);
                        
                if (children[c - Ndata][1] < Ndata):
                    print("single sample cluster 2")
                    mol_ind_2 = children[c - Ndata][1];
                    center_cluster_2 = data[mol_ind_2, :];
                else:
                    mol_ind_2 = MolDict[children[c - Ndata][1]];
                    center_cluster_2 = get_class_center(data[mol_ind_2, :]);

                rmsd = get_RMSD(center_cluster_1, center_cluster_2);
                print(rmsd)
                        
                if rmsd >= rmsd_limit:
                    print("split")
                    childlayer += children[c-Ndata].tolist()
                else:
                    class_centers[class_counter,:] = tmp_class_center;
                    class_labels[MolDict[c]] = class_counter;
                    class_counter = class_counter + 1;
                        
            else:
                class_centers[class_counter, :] = data[c, :];
                class_labels[c] = class_counter;
                class_counter = class_counter + 1;
                        
        currlayer = childlayer
    
    num_clusters = (np.unique(class_labels)).size;
    class_centers = class_centers[:num_clusters, :];
    print("Number of clusters found: " + repr(num_clusters));
    
    return class_centers, class_labels

