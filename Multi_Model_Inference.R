library(bio3d)
library(tools)
library(umap)
library(tsne)
library(ggplot2)
library(ggbiplot)
library(dbscan)
library(gridExtra)
library(matrixStats)

#**************************
#****set some paramters****
#**************************
num_neighbours = 20;
num_clusters = 5;
chain = "A";
C_alpha = TRUE;
#**************************
#**************************

#read the pdbs
myFileNames = list.files(path=".", pattern=".pdb")
main_dir = getwd()

#now get the MolProbity scores from the filename
tmp = file_path_sans_ext(myFileNames)
tmp = strsplit(tmp, "_")
molProbityScores = matrix(unlist(tmp), ncol=3, byrow=TRUE)
molProbityScores = as.numeric(molProbityScores[,3])
medianMolProb = median(molProbityScores)
subset = molProbityScores;
subset[subset>medianMolProb] = FALSE;
subset[subset != FALSE] = TRUE;
#myFileNames = myFileNames[subset==1];

#align all the pdbs
num_structures = length(myFileNames)
pdbs = pdbaln(myFileNames, fit=TRUE)

#read all atoms
xyz <- NULL
if(C_alpha==FALSE){
  for(i in myFileNames) {
    print(i)
    data = read.pdb(i);
    
    row = c()
    for(i in (1:length(data$atom$x[(data$atom$chain == chain)]))){
      row = c(row, data$atom$x[i], data$atom$y[i], data$atom$z[i])
    }
    xyz = rbind(xyz, row)
    chain_A_pdb = trim.pdb(data, atom.select(data, chain=chain))
  }
}
#get number of chains
if(chain ==""){
  chain_ids = unique(pdbs$chain[myFileNames[1], ])
}else{
  chain_ids = c(chain)
}

for(chain_index in chain_ids){
  
  tmp_dir = paste("Clustering_chain_", chain_index, sep="")
  dir.create(tmp_dir)
  setwd(tmp_dir)
  unlink('*')
  
  trimmed_pdbs = trim.pdbs(pdbs, col.inds = which((pdbs$chain[myFileNames[1], ] == chain_index)))
  
  if(C_alpha){
    xyz = (data.frame(trimmed_pdbs$xyz))
  }
  #xyz = xyz[1:400,]
  
  #******************************
  #******* do clustering ********
  #******************************
  
  #do clustering
  clusters = kmeans(xyz, centers=num_clusters)
  
  for(cluster_ind in c(1:num_clusters)){
    print(paste("size of cluster ", cluster_ind, ": ", clusters$size[cluster_ind]/sum(clusters$size)*100, "%",  sep=""))
  }
  
  #******************************
  #********** do PCA ************
  #******************************
  pc.xray <- pca(trimmed_pdbs, use.svd=TRUE)
  #plot(pc.xray)
  mktrj(pc.xray, pc=1, file = "PC1.pdb")
  mktrj(pc.xray, pc=2, file = "PC2.pdb")
  
  pca_R = prcomp(xyz)
  prediction = predict(pca_R, xyz)
  
  #get proporitons of explained variance
  var_exp = rep(0, 20)
  var_cum = rep(0, 20)
  for(i in 1:20){
    var_exp[i] = round(pca_R$sdev[i]^2/sum((pca_R$sdev)^2), digits=3)
    var_cum[i] = round(sum(pca_R$sdev[1:i]^2)/sum((pca_R$sdev)^2), digits=3)
  }
  var_data <- data.frame((1:20), var_exp, var_cum)
  
  p1 <- ggplot(data=data.frame(prediction), aes(x = PC1, y=PC2, col = factor(clusters$cluster))) +
    geom_point(size=1) + ggtitle("PCA plot") + labs(y= paste("PC2"), x = paste("PC1")) + 
    scale_fill_discrete(name="", labels=c("Control", "Treatment 1", "Treatment 2")) + theme(legend.title = element_blank()) 
  
  p2 <- ggplot(var_data, aes(x = X.1.20., y=100*var_exp)) + geom_line() + geom_point() + 
    labs(x="Principal component", y = "Proportion of variance [%]") + ggtitle("Explained variances") +
    geom_line(aes(x = X.1.20., y=100*var_exp, col='variance')) + geom_point(aes(x = X.1.20., y=100*var_exp, col='variance')) +
    geom_line(aes(x = X.1.20., y=100*var_cum, col='cum. var.')) + geom_point(aes(x = X.1.20., y=100*var_cum, col='cum. var.')) +
    theme(legend.title = element_blank()) + theme(legend.position = c(0.8, 0.5))
  
  #******************************
  #********* do UMAPS ***********
  #******************************
  set.seed(1)
  custom.settings = umap.defaults
  custom.settings$n_neighbors = num_neighbours
  #custom.settings$random_state = rnorm(1)*10000
  
  embedding = data.frame(umap(xyz, config = custom.settings)$layout)
  
  p3 <- ggplot(data=embedding, aes(x = X1, y=X2)) + ggtitle("2D embedding") + 
    geom_point(size=1) + labs(y="UMAP2", x ="UMAP1")
  
  p4 <- ggplot(data=embedding, aes(x = X1, y=X2, col = factor(clusters$cluster))) +
    geom_point(size=1) + labs(y="UMAP2", x ="UMAP1") + scale_colour_discrete("Cluster") + 
    ggtitle("UMAP embedding") + theme(legend.title = element_blank()) 
  
  #******************************
  #********* do t-SNE ***********
  #******************************
  tSNE = data.frame(tsne(xyz, perplexity = num_neighbours, whiten=FALSE))
  
  p3 <- ggplot(data=tSNE, aes(x = X1, y=X2, col = factor(clusters$cluster))) +
    geom_point(size=1) + labs(y="tSNE2", x ="tSNE1") + scale_colour_discrete("Cluster") + 
    ggtitle("tSNE embedding") + theme(legend.title = element_blank())
  
  #************************************
  #******* probe TMV structures ******* 
  #************************************
  probe_models = clusters$cluster
  probe_models[] = 0
  probe_models[1] = 1
  probe_models[2] = 2

  ggplot(data=embedding, aes(x = X1, y=X2, col = factor(probe_models))) +
    geom_point(size=1) + labs(y="UMAP2", x ="UMAP1") + scale_colour_manual(values = c("grey", "blue", "red"))
  ggsave("probed_TMV.jpg", dpi=300)
  
  #******************************
  #*** write output pdbs/pdfs ***
  #******************************
  
  cluster_ids = unique(clusters$cluster) 
  
  for(cluster_ind in cluster_ids){
    
    #write all structures in the respective cluster
    filename = paste("cluster_", cluster_ind, ".pdb", sep = "")
    write.pdb(xyz=as.matrix(xyz[(clusters$cluster==cluster_ind),]), file=filename)
    
    #get structure closest to centroid
    min_dist = 10^20;
    center_index = 0;
    for(structure_index in as.numeric(which(clusters$cluster==cluster_ind))){
      dist = norm(as.numeric(xyz[structure_index,]) - clusters$centers[cluster_ind,], type="2")
      if(dist<min_dist){
        min_dist = dist
        center_index = structure_index;
      }
    }
    
    #write mean structure of cluster
    filename = paste("center_cluster", cluster_ind, ".pdb", sep = "")
    if(C_alpha){
      write.pdb(xyz=as.numeric(xyz[center_index,]), file=filename) 
    }else{
      write.pdb(xyz=as.numeric(xyz[center_index,]), file=filename, pdb=chain_A_pdb) 
    }
  }
  
  pdf('clustering.pdf', width = 7, height = 5)
  grid.arrange(p1, p2, p3, p4, nrow=2)
  dev.off()
  
  setwd(main_dir)
}
