import numpy as np
import math
import Bio.PDB
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import ticker
import umap;
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import warnings
import random
import os
import gc


class MultiModel:

    coordinates = [];
    coord_array = [];
    res_id = [];
    tsne_embedding = [];
    pca_embedding = [];
    explained_variances = [];
    umap_embedding = [];
    umap_model = [];
    class_labels = [];
    good_classes = [];
    junk_classes = [];
    class_centers = [];
    class_size = [];
    abs_class_size = [];
    coeff_logReg = [];
    cv_score = 0;

    #************************************
    def read_pdbs(self, filenames, align=True, CA=False, chain=""):

        np.random.seed(100);
        print("Loading PDB files ...");

        for tmp_file in filenames:

            #print("Loading file " + tmp_file);

            if chain == "":
                tmp_struc = Bio.PDB.PDBParser().get_structure('hurz', tmp_file)[0];
            else:
                tmp_struc = Bio.PDB.PDBParser().get_structure('hurz', tmp_file)[0][chain];

            self.coordinates.append(tmp_struc);
            #free memory
        del tmp_struc;
        gc.collect();

        #align the structures
        if align:
            print("Aligning the atomic models based on C-alpha positions ...");
            ref_model = self.coordinates[0];
            num_structures = len(self.coordinates);

            for model_ind in range(num_structures):

                alt_model = self.coordinates[model_ind];

                # Build paired lists of c-alpha atoms, ref_atoms and alt_atoms
                ref_atoms = [];
                alt_atoms = [];

                if chain == "":
                    #analyse all chains
                    for (ref_chain, alt_chain) in zip(ref_model, alt_model):

                        for ref_res, alt_res in zip(ref_chain, alt_chain):

                            # CA = alpha carbon
                            aa = True;
                            try:
                                ref_res['CA'];
                            except:
                                aa = False;

                            if aa:
                                ref_atoms.append(ref_res['CA']);
                                alt_atoms.append(alt_res['CA']);

                else:
                    #subset the desired chain
                    #ref_chain = ref_model[chain];
                    #alt_chain = alt_model[chain];

                    ref_chain = ref_model;
                    alt_chain = alt_model;

                    for ref_res, alt_res in zip(ref_chain, alt_chain):

                        aa = True;
                        try:
                            ref_res['CA'];
                        except:
                            aa = False;

                        if aa:
                            ref_atoms.append(ref_res['CA']);
                            alt_atoms.append(alt_res['CA']);

                # Align these paired atom lists:
                super_imposer = Bio.PDB.Superimposer()
                super_imposer.set_atoms(ref_atoms, alt_atoms)

                if (model_ind == 0):
                    # Check for self/self get zero RMS, zero translation
                    # and identity matrix for the rotation.
                    assert np.abs(super_imposer.rms) < 0.0000001
                    assert np.max(np.abs(super_imposer.rotran[1])) < 0.000001
                    assert np.max(np.abs(super_imposer.rotran[0]) - np.identity(3)) < 0.000001
                else:
                    # Update the structure by moving all the atoms in
                    # this model (not just the ones used for the alignment)
                    alt_model = 0;
                    super_imposer.apply( self.coordinates[model_ind].get_atoms())


        #make vector representation of coordinates
        self.coord_array = [];
        for tmp_model in self.coordinates:

            tmp_coord = [];
            tmp_res_id = [];

            if chain == "":
                for tmp_chain in tmp_model:

                    chain_id = tmp_chain.id;

                    for tmp_res in tmp_chain:
                        res_id = tmp_res.id;
                        res_name = repr(chain_id) + repr(res_id[1]);
                        if(CA):
                            try:#if CA atom exists, append it
                                tmp_atom = tmp_res['CA'];
                                tmp_coord.append(tmp_atom.get_coord());
                                tmp_res_id.append([res_name, res_name, res_name])
                            except:
                                pass
                        else:
                            #append all atoms
                            for tmp_atom in tmp_res:
                                tmp_coord.append( tmp_atom.get_coord());
                                tmp_res_id.append([res_name, res_name, res_name])
            else:
                tmp_chain = tmp_model;

                chain_id = tmp_chain.id;

                for tmp_res in tmp_chain:
                    res_id = tmp_res.id;
                    res_name = repr(chain_id) + repr(res_id[1]);
                    if (CA):
                        try:  # if CA atom exists, append it
                            tmp_atom = tmp_res['CA'];
                            tmp_coord.append(tmp_atom.get_coord());
                            tmp_res_id.append([res_name, res_name, res_name])
                        except:
                            pass
                    else:
                        # append all atoms
                        for tmp_atom in tmp_res:
                            tmp_coord.append(tmp_atom.get_coord());
                            tmp_res_id.append([res_name, res_name, res_name])


            tmp_coord = np.asarray(tmp_coord).flatten();
            self.coord_array.append(tmp_coord);

            tmp_res_id = np.asarray(tmp_res_id).flatten();
            self.res_id= tmp_res_id;

        self.coord_array = np.asarray(self.coord_array);
        self.res_id = np.asarray(self.res_id);

    #**************************************
    def do_pca_embedding(self):

        pca = PCA();
        pca.fit(self.coord_array);
        self.explained_variances = pca.explained_variance_ratio_ * 100;

        pca = PCA(n_components=2);
        pca.fit(self.coord_array);
        self.pca_embedding = pca.fit_transform(self.coord_array);


    #**************************************
    def do_tsne_embedding(self, num_neighbors):

        self.tsne_embedding = TSNE(n_components=2, perplexity=num_neighbors, init="pca").fit_transform(self.coord_array);


    #**************************************
    def do_umap_embedding(self, num_neighbors):

        self.umap_model = umap.UMAP(n_neighbors=num_neighbors).fit(self.coord_array);
        self.umap_embedding = self.umap_model.transform(self.coord_array);


    #***************************************
    def do_classification(self, num_classes, reduced=False):

        print("Classifying atomic models ...");

        if reduced:
            classes = KMeans(n_clusters=num_classes, random_state=0).fit(self.umap_embedding);
        else:
            classes = KMeans(n_clusters=num_classes, random_state=0).fit(self.coord_array);

        self.class_labels = classes.labels_;
        self.class_centers = classes.cluster_centers_;

        #get relative class sizes
        _, self.abs_class_size = np.unique(self.class_labels, return_counts=True);
        self.class_size = self.abs_class_size/float(np.sum(self.abs_class_size));

        # sort classes according to size
        sorted_classes = np.flip(np.argsort(self.class_size));
        self.class_size = self.class_size[sorted_classes];
        self.class_centers = self.class_centers[sorted_classes];
        self.abs_class_size = self.abs_class_size[sorted_classes];

        for tmp_sample in range(self.class_labels.size):
            self.class_labels[tmp_sample] = sorted_classes[self.class_labels[tmp_sample]];

        for class_ind in range(num_classes):
            print("Relative size of class {}: {:.2f}%.".format(class_ind, self.class_size[class_ind]*100));


        #take all classes with size >=10
        self.good_classes = np.argwhere(self.abs_class_size>=10).flatten();
        self.junk_classes = np.argwhere(self.abs_class_size<10).flatten();



        #self.class_labels[np.isin(self.class_labels, junk_classes)] = -1;

        #new_class = 0;
        #for tmp_class in final_classes:
        #    self.class_labels[self.class_labels == tmp_class] = new_class;
        #    new_class = new_class + 1;

        #self.class_size = self.class_size[final_classes];
        #self.abs_class_size = self.abs_class_size[final_classes];
        #self.class_centers = self.class_centers[final_classes];
        #self.coord_array = self.coord_array[np.isin(self.class_labels, final_classes)];
        #print(np.arange(self.class_labels.size)[np.isin(self.class_labels, final_classes)]);
        #self.coordinates = [self.coordinates[i] for i in np.arange(self.class_labels.size)[np.isin(self.class_labels, final_classes)]];
        #self.class_labels = self.class_labels[np.isin(self.class_labels, final_classes)];

    #***************************************
    def SVM(self):

        print("Training support vector machine ...");

        warnings.filterwarnings("ignore");
        clf = svm.SVC(kernel='rbf');
        #clf.fit(self.coord_array, self.class_labels);

        scores = cross_val_score(clf, self.coord_array[np.isin(self.class_labels, self.good_classes)], self.class_labels[np.isin(self.class_labels, self.good_classes)], cv=5);
        self.cv_score = scores.mean()*100;
        if scores.mean() < 0.95:
            print("Cross-validation score of cluster assignment: {:.2f} %".format(scores.mean() * 100));
            print("WARNING: Data is likely overclustered! Cross-validation score < 95%");
        else:
            print("Cross-validation score of cluster assignment: {:.2f} %".format(scores.mean() * 100));

    #***************************************
    def logistic_regression(self):

        print("Training of logistic regression ...");

        warnings.filterwarnings("ignore");
        clf = LogisticRegression(random_state=0, multi_class='multinomial', solver='lbfgs', max_iter=1000);
        logistic_regression = clf.fit(self.coord_array, self.class_labels);

        #save the regression coefficients
        self.coeff_logReg = logistic_regression.coef_;

        scores = cross_val_score(clf, self.coord_array[np.isin(self.class_labels, self.good_classes)], self.class_labels[np.isin(self.class_labels, self.good_classes)], cv=5);
        self.cv_score = scores.mean()*100;
        if scores.mean() < 0.95:
            print("Cross-validation score of cluster assignment: {:.2f} %".format(scores.mean() * 100));
            print("WARNING: Data is likely overclustered! Cross-validation score < 95%");
        else:
            print("Cross-validation score of cluster assignment: {:.2f} %".format(scores.mean()*100));

    #***************************************
    def plot_coeffs(self):

        num_classes = self.good_classes.size;

        fig = plt.figure()
        fig.suptitle("Cross-validation score of cluster assignment: {:.2f} %".format(self.cv_score));
        gs = gridspec.GridSpec(math.ceil(num_classes/2), 2, wspace=0.5, hspace=1.0);

        #get unique residue identifiers
        _, idx = np.unique(self.res_id, return_index=True);
        unique_res_ids = self.res_id[np.sort(idx)];

        # average the logistic regression coefficients of each residue
        coeffs_avg = np.zeros((num_classes, unique_res_ids.size));
        i = 0;
        for tmp_res in unique_res_ids:
            coeffs_avg[:, i] = np.mean(self.coeff_logReg[:num_classes, np.nonzero(self.res_id==tmp_res)], axis=2).flatten();
            i = i+1;

        #generate the plots
        tmp_class_ind = 0;
        for x_ind in range(math.ceil(num_classes/2)):
            for y_ind in range(2):

                class_ind = self.good_classes[tmp_class_ind];

                res_ids = range(coeffs_avg.shape[1]);
                ax = plt.subplot(gs[x_ind, y_ind]);
                ax.set_title('Class ' + repr(class_ind));
                ax.plot(unique_res_ids, np.abs(coeffs_avg[class_ind,:]));
                xticks = ticker.MaxNLocator(10);
                ax.xaxis.set_major_locator(xticks);
                ax.tick_params(labelsize=4)

                #label the peaks
                sorted_indices = np.argsort(np.abs(coeffs_avg[class_ind, :]));
                sorted_indices = sorted_indices[::-1]; #reverse order
                for tmp_ind in sorted_indices[0:5]:
                    ax.annotate(unique_res_ids[tmp_ind], (tmp_ind, np.abs(coeffs_avg[class_ind, tmp_ind])), ha='center', fontsize=5);

                tmp_class_ind = tmp_class_ind + 1;
                if tmp_class_ind > (num_classes-1):
                    break;

        plt.savefig("Regression_Coefficients.pdf",dpi=600);
        plt.close();

    #***************************************
    def make_plots(self):

        colors = "jet"
        num_classes = np.unique(self.class_labels).size;
        tmp_label = np.unique(self.class_labels);
        label = [""]*num_classes;
        for class_ind in range(num_classes):
            label[class_ind] = "{}\n({:.0f}%)".format(tmp_label[class_ind], self.class_size[class_ind]*100);

        # make diagnostics plot
        plt.rc('xtick', labelsize=8);  # fontsize of the tick labels
        plt.rc('ytick', labelsize=8);  # fontsize of the tick labels
        gs = gridspec.GridSpec(2, 2, wspace=0.5, hspace=0.5);


        #plot pca
        ax1 = plt.subplot(gs[0, 0]);
        scatter = ax1.scatter(self.pca_embedding[:,0], self.pca_embedding[:,1], c=self.class_labels, s=4.0, cmap=colors);
        ax1.set_title('PCA plot');
        ax1.set_xlabel('PC 1');
        ax1.set_ylabel('PC 2');
        ax1.legend(handles=scatter.legend_elements()[0], labels=label, title='Class\n(rel.size)', fontsize=4, title_fontsize=4,
                   bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.);

        #plot explained variances
        ax2 = plt.subplot(gs[0, 1]);
        ax2.plot(range(1, 21, 1), self.explained_variances[0:20], linewidth=2, label="variance");
        ax2.plot(range(1, 21, 1), np.cumsum(self.explained_variances[0:20]), linewidth=2, label="cumulative\nvariance")
        ax2.set_xticks([1,2,3,4,5,10,15,20]);
        ax2.set_ylim(0,100);
        ax2.set_xlabel('Principal Component');
        ax2.set_ylabel('Explained variance [%]');
        ax2.set_title('Explained variances');
        ax2.legend( title='', fontsize=4, title_fontsize=4,
                   bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.);


        #plot t-SNE
        ax3 = plt.subplot(gs[1, 0]);
        scatter = ax3.scatter(self.tsne_embedding[:,0], self.tsne_embedding[:,1], c=self.class_labels, s=4.0, cmap=colors);
        ax3.set_title('t-SNE plot');
        ax3.set_xlabel('t-SNE 1');
        ax3.set_ylabel('t-SNE 2');
        ax3.legend(handles=scatter.legend_elements()[0], labels=label, title='Class\n(rel.size)', fontsize=4, title_fontsize=4,
                   bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.);

        #plot umap
        ax4 = plt.subplot(gs[1, 1]);
        scatter = ax4.scatter(self.umap_embedding[:,0], self.umap_embedding[:,1], c=self.class_labels, s=4.0, cmap=colors);
        ax4.set_title('UMAP plot');
        ax4.set_xlabel('UMAP 1');
        ax4.set_ylabel('UMAP 2');
        ax4.legend(handles=scatter.legend_elements()[0], labels=label, title='Class\n(rel.size)', fontsize=4, title_fontsize=4,
                   bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.);

        plt.savefig("Model_Classification.pdf", dpi=300);
        plt.close();

    #***************************************
    def probe_tmv(self):

        colors = "jet"
        umap_embedding = self.umap_model.transform(self.coord_array);

        TMVLabels  =self.classes.labels_;
        TMVLabels[:] = 0;
        TMVLabels[200] = 1;
        TMVLabels[201] = 2;

        plt.scatter(umap_embedding[:,0], umap_embedding[:,1], c=TMVLabels, s=50.0, cmap=colors);

        plt.savefig("TMV_probed.pdf", dpi=300);

    #***************************************
    def write_pdbs(self, filenames, reduced=True):

        print("Writing class centers as pdbs ...")

        num_classes = self.class_centers.shape[0];
        num_samples = self.class_labels.shape[0];

        for tmp_class in np.unique(self.class_labels):


            if tmp_class != -1:
                #get sample closest to center of cluster
                min_dist = 1.0*10**10;
                center_index = 0;
                for tmp_sample in range(num_samples):

                    #tmp_dist = np.sqrt(np.sum(np.square(self.coord_array[tmp_sample,:] - self.classes.cluster_centers_[tmp_class,:])));
                    if reduced:
                        tmp_dist = np.sqrt(np.sum(np.square(self.umap_embedding[tmp_sample,:] - self.class_centers[tmp_class,:])));
                    else:
                        tmp_dist = np.sqrt(np.sum(np.square(self.coord_array[tmp_sample,:] - self.class_centers[tmp_class,:])));


                    if tmp_dist < min_dist:
                        center_index = tmp_sample;
                        min_dist = tmp_dist;

                #write the structure
                io = Bio.PDB.PDBIO()
                io.set_structure(self.coordinates[center_index]);
                io.save('Center_Class' + repr(tmp_class) + '.pdb');
                del io;
                gc.collect();

            indices = np.asarray(range(self.class_labels.size));
            models_in_class = indices[self.class_labels==tmp_class];
            folder = "Class_" + repr(tmp_class);
            os.mkdir(folder);

            for tmp_model in models_in_class:

                filename = folder + "/" + os.path.basename(filenames[tmp_model]);
                io = Bio.PDB.PDBIO()
                io.set_structure(self.coordinates[tmp_model]);
                io.save(filename);

            del io;
            gc.collect();




