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
num_min_points = 20;
num_clusters = 3;
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
#xyz <- NULL
#for(i in myFileNames) {
#  print(i)
#  data = read.pdb(i);
#  xyz = rbind(xyz, c(data$atom$x[(data$atom$chain == "G")], data$atom$y[(data$atom$chain == "G")], data$atom$z[(data$atom$chain == "G")] ))
#}

#get number of chains
chain_ids = unique(pdbs$chain[myFileNames[1], ])

for(chain_index in chain_ids){
  
  tmp_dir = paste("Clustering_chain_", chain_index, sep="")
  dir.create(tmp_dir)
  setwd(tmp_dir)
  unlink('*')
  
  trimmed_pdbs = trim.pdbs(pdbs, col.inds = which((pdbs$chain[myFileNames[1], ] == chain_index)))
  
  #make cross correlation matrix
  #cij<-dccm(trimmed_pdbs$xyz)
  #plot(cij)
  
  xyz = (data.frame(trimmed_pdbs$xyz))
  
  #xyz = xyz[1:300,]
  
  #******************************
  #******* do clustering ********
  #******************************
  
  #do clustering
  #clusters = hdbscan(xyz, minPts = num_min_points)
  clusters = kmeans(xyz, centers=num_clusters)
  
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
    geom_point(size=1) + ggtitle("PCA plot") + scale_colour_discrete("Cluster") +
    labs(y= paste("PC2"), x = paste("PC1")) + theme(legend.title = element_blank())
  
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
        center_index = structure_index;
      }
    }
    
    #write mean structure of cluster
    filename = paste("center_cluster", cluster_ind, ".pdb", sep = "")
    #write.pdb(xyz=colMeans(as.matrix(xyz[(clusters$cluster==cluster_ind),])), file=filename)
    write.pdb(xyz= clusters$centers[cluster_ind,], file=filename)  
    #write.pdb(xyz=as.numeric(xyz[structure_index,]), file=filename)  
  }
  
  pdf('clustering.pdf', width = 7, height = 5)
  grid.arrange(p1, p2, p3, p4, nrow=2)
  dev.off()
  
  setwd(main_dir)
}
