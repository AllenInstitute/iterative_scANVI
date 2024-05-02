import os
import re
import json
import hashlib
import glob
import copy
import random
import scvi
import numpy as np
import pandas as pd
import seaborn as sns
import scanpy as sc
import anndata as ad
import rapids_singlecell as rsc
from datetime import datetime
from scipy import sparse as sp_sparse
from scipy import stats as sp_stats
from matplotlib import pyplot as plt
from joblib import parallel_backend
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore")

'''
Integrates and predicts labels for a query dataset iteratively using scVI and scANVI (Xu et al 2021 Mol Syst Biol)

Input arguements:
adata_query: (AnnData) Object with unknown cells

adata_ref: (AnnData) Object with reference labels

labels_keys: (list) Ordered list of labels to iteratively predict and split (e.g. ["class", "subclass", "cluster"])

output_dir: (str) Location to store trained models and final results CSV

**kwargs: (dict) Passed to several functions, details below:

    layer: (None or str, default None) None if unnormalized counts are in AnnData.X, else a str where they are stored in AnnData.layers
    
    batch_key: (None or str, default None) Name of the batch variable pass to scVI and scANVI (e.g. "donor_name")
    
    categorical_covariate_keys: (list) List of categorical covariates to pass to scVI and scANVI (e.g. ["donor_name", "sex"])
    
    continuous_covariate_keys: (list) List of continuous covariates to pass to scVI and scANVI (e.g. ["n_genes"])
    
    use_hvg: (bool, default True) Whether to calculate and include highly variable genes from the reference dataset to scVI and scANVI
    
    use_de: (bool, default True) Whether to calculate and include differentially genes from the reference dataset to scVI and scANVI
    
    n_top_genes: (int or list, default 2000) Number of highly variable genes to pass in each iteration. Not used if use_hvg == False
    
    n_downsample_ref: (int or list, default 1000) Number of cells to downsample reference groups too in each iteration when 
    calculcating differentially expressed genes
    
    n_ref_genes: (int or list, default 500): Number of differentially expressed genes per group to pass in each iteration from the 
    reference dataset to scVI and scANVI
    
    user_genes: (list, default None): List of either lists (to use the same genes across all types within a taxonomy level) or dicts (to specify difference gene lists within each type in a taxonomy level) of user defined genes to use with the model

    max_epochs_scVI: (int or list, default 200) Number of epochs to train scVI for in each iteration
    
    max_epochs_scANVI: (int or list, default 20) Number of epochs to train scANVI for in each iteration 
    
    min_accuracy: (float, default 0.85) Minimum accuracy for label self-projection needed to avoid an accuracy flag in cleanup (See output).
    
    plot_confusion: (bool, default True) Whether to plot the confusion matrix after scANVI label prediction
    
    plot_latent_space: (bool, default False) Whether to plot the variables used in the model on a UMAP representation of the latent space

    save_latent_space: (bool, default False) Whether to save the latent space as an numpy matrix in the model folder

    scVI_model_args: (dict, default {"n_layer": 2}) kwargs passed to scvi.model.SCVI
    
    scANVI_model_args: (dict, default {}) kwargs passed to scvi.model.SCANVI.from_scvi_model

Outputs:
scVI models in output_dir/scVI_models

scANVI models in output_dir/scANVI_models

CSV named "iterative_scANVI_results.<date>.csv" in output_dir with cells as rows and the following columns:

    <Label>_stash (str) Original labels from reference and query datasets
    
    <Label> (str) Original labels from reference and query datasets, coerced to "Unknown" if its predicted label disagreed with ground truth
    
    <Label>_scANVI: (str) Predicted type for each level of the hierarchy (if [labels_keys] was ["class", "subclass"] class_scANVI
    and subclass_scANVI would exist)
    
    <Label>_conf_scANVI: (float) Probability of the prediction
    
    <All Possible Label Values>: (float) Probability a cell is that label (e.g. glia, excitatory, inhibitory or Astro, Endo, VLMC, etc)

    
Example Usage:
iterative_scANVI_kwargs = {
    "categorical_covariate_keys": ["donor_name"]
    "continuous_covariate_keys": ["n_genes"]
    "n_top_genes": [5000, 2000, 2000]
}

iteratively_map(
    adata_query, 
    adata_ref,
    labels_keys=["class", "subclass", "cluster"]
    output_dir=os.path.join("scANVI_output"), 
    **iterative_scANVI_kwargs
)

'''

def iteratively_map(adata_query, adata_ref, labels_keys, output_dir, **kwargs):
    
    default_kwargs = {
        "layer": None,
        "batch_key": None,
        "categorical_covariate_keys": None,
        "continuous_covariate_keys": None,
        "use_hvg": True,
        "use_de": True,
        "n_top_genes": 2000,
        "n_downsample_ref": 1000,
        "min_ref_cells": 15,
        "n_ref_genes": 500,
        "user_genes": None,
        "max_epochs_scVI": 200,
        "max_epochs_scANVI": 20,
        "scVI_model_args": {"n_layers": 2},
        "scANVI_model_args": {"n_layers": 2},
        "save_latent_space": False,
        "min_accuracy": 0.85,
        "plot_confusion": True,
        "plot_latent_space": False,
        "add_vars_to_plot": None,

    }
    
    kwargs = {**default_kwargs, **kwargs}
    
    layer = kwargs["layer"]
    batch_key = kwargs["batch_key"]
    categorical_covariate_keys = kwargs["categorical_covariate_keys"]
    continuous_covariate_keys = kwargs["continuous_covariate_keys"]
    use_hvg = kwargs["use_hvg"]
    use_de = kwargs["use_de"]
    n_top_genes = kwargs["n_top_genes"]
    min_ref_cells = kwargs["min_ref_cells"]
    n_downsample_ref = kwargs["n_downsample_ref"]
    n_ref_genes = kwargs["n_ref_genes"]
    user_genes = kwargs["user_genes"]
    max_epochs_scVI = kwargs["max_epochs_scVI"]
    max_epochs_scANVI = kwargs["max_epochs_scANVI"]
    scVI_model_args = kwargs["scVI_model_args"]
    scANVI_model_args = kwargs["scANVI_model_args"]
    save_latent_space = kwargs["save_latent_space"]
    min_accuracy = kwargs["min_accuracy"]
    plot_confusion = kwargs["plot_confusion"]
    plot_latent_space = kwargs["plot_latent_space"]
    add_vars_to_plot = kwargs["add_vars_to_plot"]

    
    if isinstance(adata_query, ad.AnnData) == False or isinstance(adata_ref, ad.AnnData) == False:
        raise TypeError("One or more of the AnnData objects have an incorrect type,")

    if adata_query.shape[1] != adata_ref.shape[1] or all(adata_query.var_names != adata_ref.var_names) != True:
        common_labels = np.intersect1d(adata_query.var_names, adata_ref.var_names)
        
        if len(common_labels) == 0:
            raise IndexError("Reference and query AnnData objects have different shapes and no overlapping var_names.")
        adata_ref = adata_ref[:, common_labels].copy()
        adata_query = adata_query[:, common_labels].copy()
        warnings.warn("Reference and query AnnData objects have different shapes, using " + str(len(common_labels)) + " common var_names. This may have a deterimental effect on model performance.")

    if len(labels_keys) == 0:
        raise ValueError("You must specify at least 1 label to map to.")

    if isinstance(labels_keys, list) == False:
        raise TypeError("labels_keys must be a list.")
        
    if all([i in adata_ref.obs.columns for i in labels_keys]) == False:
        raise KeyError("One or more labels_keys do not exist in the reference AnnData object.")

    for i in labels_keys:
        adata_ref.obs[i] = adata_ref.obs[i].astype("category")
        if batch_key != None:
            if len(np.intersect1d(adata_ref.obs[i].cat.categories, batch_key)) > 0:
                raise ValueError("The batch key cannot also be the name of a value in any labels_key.")

        if categorical_covariate_keys != None:
            if len(np.intersect1d(adata_ref.obs[i].cat.categories, categorical_covariate_keys)) > 0:
                raise ValueError("The categorical_covariate_keys cannot also be the name of a value in any labels_key.")

        if continuous_covariate_keys != None:
            if len(np.intersect1d(adata_ref.obs[i].cat.categories, continuous_covariate_keys)) > 0:
                raise ValueError("The continuous_covariate_keys cannot also be the name of a value in any labels_key.")
    
    if layer != None:
        if layer not in adata_query.layers.keys() or layer not in adata_ref.layers.keys():
            raise KeyError("Layer " + layer + " does not exist in both AnnData objects.")
        
        if isinstance(adata_query.layers[layer], sp_sparse.csr_matrix) == False:
            adata_query.layers[layer] = sp_sparse.csr_matrix(adata_query.layers[layer])
            warnings.warn("Counts matrix in the query anndata was stored as a dense matrix, converting to scipy.sparse.csr_matrix.")

        if isinstance(adata_ref.layers[layer], sp_sparse.csr_matrix) == False:
            adata_ref.layers[layer] = sp_sparse.csr_matrix(adata_ref.layers[layer])
            warnings.warn("Counts matrix in the reference anndata was stored as a dense matrix, converting to scipy.sparse.csr_matrix.")

        if len(adata_query.layers[layer].data) == 0 or len(adata_ref.layers[layer].data) == 0:
            raise ValueError("There are no non-zero values in one of the anndata counts matrices.")
        
        if all([i.is_integer() for i in adata_query.layers[layer].data]) == False or all([i.is_integer() for i in adata_ref.layers[layer].data]) == False:
            raise TypeError("One of the anndata counts matrices has non-integer values. This is often caused by data normalization. scVI and scANVI require the raw count matrix.")
    
   
    else:
        if isinstance(adata_query.X, sp_sparse.csr_matrix) == False:
            adata_query.X = sp_sparse.csr_matrix(adata_query.X)
            warnings.warn("Counts matrix was stored as a dense matrix, converting to scipy.sparse.csr_matrix.")

        if isinstance(adata_ref.X, sp_sparse.csr_matrix) == False:
            adata_ref.X = sp_sparse.csr_matrix(adata_ref.X)
            warnings.warn("Counts matrix was stored as a dense matrix, converting to scipy.sparse.csr_matrix.")

        if len(adata_query.X.data) == 0 or len(adata_ref.X.data) == 0:
            raise ValueError("There are no non-zero values in one of the anndata counts matrices.")
        
        if all([i.is_integer() for i in adata_query.X.data]) == False or all([i.is_integer() for i in adata_ref.X.data]) == False:
            raise TypeError("One of the anndata counts matrices has non-integer values. This is often caused by data normalization. scVI and scANVI require the raw count matrix.")
        
    if batch_key != None:
        if batch_key not in adata_query.obs.columns or batch_key not in adata_ref.obs.columns:
            raise KeyError("Batch key != in both AnnData objects.")
    try:
        if all([i in adata_query.obs.columns for i in categorical_covariate_keys]) == False or all([i in adata_ref.obs.columns for i in categorical_covariate_keys]) == False:
            raise KeyError("One or more categorical covariates do not exist in both AnnData objects.")
    except:
        pass
    
    try:    
        if all([i in adata_query.obs.columns for i in continuous_covariate_keys]) == False or all([i in adata_ref.obs.columns for i in continuous_covariate_keys]) == False:
            raise KeyError("One or more continuous covariates do not exist in both AnnData objects.")
    except:
        pass

    try:    
        if all([i in adata_query.obs.columns for i in add_vars_to_plot]) == False or all([i in adata_ref.obs.columns for i in add_vars_to_plot]) == False:
            raise KeyError("One or more additional variables to plot do not exist in both AnnData objects.")
    except:
        pass
                          
    for i in [use_hvg, use_de, n_top_genes, n_downsample_ref, min_ref_cells, n_ref_genes, user_genes, max_epochs_scVI, max_epochs_scANVI]:
        if isinstance(i, bool) or isinstance(i, str) or isinstance(i, int) or i is None:
            i = [i]
        if len(i) != 1 and len(i) != len(labels_keys):
            raise ValueError(str(i) + " should be either a single value used in each iteration or a list of values of equal length to labels_keys.")
    
    print(str(datetime.now()) + " -- All validation steps completed.")

    query_vars = []
    for i in [categorical_covariate_keys, continuous_covariate_keys, batch_key, add_vars_to_plot]:
        if i != None:
            if isinstance(i, str) == True:
                query_vars.append(i)
            else:
                query_vars.extend(i)

    adata_query.obs = adata_query.obs.loc[:, query_vars].copy()

    ref_vars = query_vars + labels_keys
    adata_ref.obs = adata_ref.obs.loc[:, ref_vars].copy()

    adata = ad.concat([adata_query, adata_ref], join="outer", merge="unique")
    del adata_query

    adata.obs_names = [str(i) for i in adata.obs_names]
    adata.ref = [str(i) for i in adata_ref.obs_names]

    try:
        iter(min_ref_cells)
        _min_ref_cells = min_ref_cells[-1]
    except:
        _min_ref_cells = min_ref_cells

    try:
        iter(n_downsample_ref)
        _n_downsample_ref = n_downsample_ref[-1]
    except:
        _n_downsample_ref = n_downsample_ref


    ref_counts = adata_ref.obs[labels_keys[-1]].value_counts()
    adata_ref = adata_ref[~(adata_ref.obs[labels_keys[-1]].isin(ref_counts[ref_counts < _min_ref_cells].index))]
            
    cells = []
    for i in adata_ref.obs[labels_keys[-1]].cat.categories:
        tmp_cells = adata_ref[adata_ref.obs[labels_keys[-1]] == i].obs_names.to_list()
        
        if len(tmp_cells) > n_downsample_ref:
            cells = cells + random.sample(tmp_cells, k=_n_downsample_ref)

        else:
            cells.extend(tmp_cells)

    adata_ref = adata_ref[cells].copy()

    print(str(datetime.now()) + " -- Finished creating merged and downsampled AnnData objects.") 
    
    
    for i,j in enumerate(labels_keys):
        get_model_genes_kwargs = {
            "layer": layer,
            "groupby": j,
            "use_hvg": use_hvg[i] if isinstance(use_hvg, list) else use_hvg,
            "use_de": use_de[i] if isinstance(use_de, list) else use_de,
            "n_top_genes": n_top_genes[i] if isinstance(n_top_genes, list) else n_top_genes,
            "n_downsample_ref": n_downsample_ref[i] if isinstance(n_downsample_ref, list) else n_downsample_ref,
            "min_ref_cells": min_ref_cells[i] if isinstance(min_ref_cells, list) else min_ref_cells,
            "n_ref_genes": n_ref_genes[i] if isinstance(n_ref_genes, list) else n_ref_genes,
            "user_genes": user_genes[i] if isinstance(user_genes, list) else None
        }
        run_scVI_kwargs = {
            "layer": layer,
            "max_epochs_scVI": max_epochs_scVI[i] if isinstance(max_epochs_scVI, list) else max_epochs_scVI,
            "batch_key": batch_key,
            "categorical_covariate_keys": categorical_covariate_keys,
            "continuous_covariate_keys": continuous_covariate_keys,
            "scVI_model_args": scVI_model_args
        }
        run_scANVI_kwargs = {
            "layer": layer,
            "max_epochs_scANVI": max_epochs_scANVI[i] if isinstance(max_epochs_scANVI, list) else max_epochs_scANVI,
            "batch_key": batch_key,
            "categorical_covariate_keys": categorical_covariate_keys,
            "continuous_covariate_keys": continuous_covariate_keys,
            "labels_key": j,
            "scANVI_model_args": scANVI_model_args
        }
                
        adata.obs[j] = adata.obs[j].astype('object')
        adata.obs.loc[adata.obs[j].isna(), j] = "Unknown"
        adata.obs[j] = adata.obs[j].astype('category')
        adata.obs[j + "_stash"] = adata.obs[j].copy()

        # To do: Unify the if i == 0 and else: blocks.  
        if i == 0:
            print(str(datetime.now()) + " --- Training model predicting " + j + " on entire dataset")
            model_name = hashlib.md5(str(json.dumps({**get_model_genes_kwargs, **run_scVI_kwargs})).replace("/", " ").encode()).hexdigest()
            label_model_name = hashlib.md5(str(json.dumps({**get_model_genes_kwargs, **run_scANVI_kwargs})).replace("/", " ").encode()).hexdigest()

            if os.path.exists(os.path.join(output_dir, "scANVI_models", label_model_name)) == False:
                
                if os.path.exists(os.path.join(output_dir, "scVI_models", model_name)) == False:
                    if user_genes == None:
                        markers = get_model_genes(adata_ref, **get_model_genes_kwargs)
                    else:
                        markers = user_genes[i]
                    model = run_scVI(adata[:, markers], **run_scVI_kwargs)
                    model.save(os.path.join(output_dir, "scVI_models", model_name))
                    pd.DataFrame(markers).to_csv(
                        os.path.join(output_dir, "scVI_models", model_name, "var_names.csv"),
                        index=False,
                        header=False
                    )
                    if save_latent_space == True:
                        pd.DataFrame(adata.obs_names).to_csv(
                            os.path.join(output_dir, "scVI_models", model_name, "obs_names.csv"),
                            index=False,
                            header=False
                        )
                        latent_space = model.get_latent_representation()
                        np.save(
                            file=os.path.join(output_dir, "scVI_models", model_name, "X_scVI.npy"),
                            arr=latent_space
                        )
                        del latent_space

                else:
                    markers = pd.read_csv(os.path.join(output_dir, "scVI_models", model_name, "var_names.csv"), header=None)
                    markers = markers[0].to_list()
                    model = scvi.model.SCVI.load(os.path.join(output_dir, "scVI_models", model_name), adata[:, markers].copy())
                
                label_model, probabilities = run_scANVI(adata[:, markers], model=model, **run_scANVI_kwargs)            
                label_model.save(os.path.join(output_dir, "scANVI_models", label_model_name))
                pd.DataFrame(markers).to_csv(
                    os.path.join(output_dir, "scANVI_models", label_model_name, "var_names.csv"),
                    index=False,
                    header=False
                )
                probabilities.to_csv(os.path.join(output_dir, "scANVI_models", label_model_name, "probabilities.csv"))
                
                if save_latent_space == True or plot_latent_space == True:
                    if save_latent_space == False and plot_latent_space == True:
                        warnings.warn("The scANVI model latent space is saved by default when plot_latent_space==True and save_latent_space==False.")

                    pd.DataFrame(adata.obs_names).to_csv(
                        os.path.join(output_dir, "scANVI_models", label_model_name, "obs_names.csv"),
                        index=False,
                        header=False
                    )
                    latent_space = label_model.get_latent_representation()
                    np.save(
                        file=os.path.join(output_dir, "scANVI_models", label_model_name, "X_scVI.npy"),
                        arr=latent_space
                    )
                    del latent_space

                    
            else:
                probabilities = pd.read_csv(os.path.join(output_dir, "scANVI_models", label_model_name, "probabilities.csv"), index_col=0)
            
            probabilities = probabilities.dropna(axis=1, how='all')
            probabilities = probabilities.drop([l for l in probabilities.columns if l.startswith("_")], axis=1)
            tmp = pd.concat([adata.obs, probabilities], axis=1)
            for l in [m for m in tmp.columns if m.endswith("_y")]:
                l = l.replace("_y", "")
                tmp[l + "_x"] = tmp[l + "_x"].astype("object")
                tmp[l + "_y"] = tmp[l + "_y"].astype("object")
                tmp[l] = tmp[l + "_y"].fillna(tmp[l + "_x"])
                tmp[l] = tmp[l].astype("category")
                tmp = tmp.drop([l + "_y", l + "_x"], axis=1)

            adata.obs = tmp.copy()

            conf_mat = adata.obs.groupby([j, j + "_scANVI"]).size().unstack(fill_value=0)

            if plot_confusion == True:
                try:
                    display(conf_mat)
                except:
                    print(conf_mat)

            conf_mat = conf_mat.div(conf_mat.sum(axis=1), axis=0)
            conf_mat[conf_mat.isna()] = 0
            conf_mat = conf_mat.reindex(sorted(conf_mat.columns), axis=1)
            conf_mat = conf_mat.reindex(sorted(conf_mat.index), axis=0)

            for l in conf_mat.index:
                if l == "Unknown":
                    continue
                elif l not in conf_mat.columns:
                    print("WARNING: Label " + l + " fell below accruacy threshold " + str(min_accuracy) + " on reference cells. Label was not used.")
                elif conf_mat.loc[l,l] < min_accuracy:
                    print("WARNING: Label " + l + " fell below accruacy threshold " + str(min_accuracy) + " on reference cells. Accuracy=" + str(conf_mat.loc[l,l]))

            if plot_confusion == True:
                plt.figure(figsize=(10, 10))
                ax = plt.pcolor(conf_mat)
                ax = plt.xticks(np.arange(0.5, len(conf_mat.columns), 1), conf_mat.columns, rotation=90)
                ax = plt.yticks(np.arange(0.5, len(conf_mat.index), 1), conf_mat.index)
                plt.xlabel("Predicted")
                plt.ylabel("Observed")
                plt.colorbar()
                plt.show()


            if plot_latent_space == True:
                adata_min = ad.AnnData(
                        X=sp_sparse.csr_matrix(np.zeros((adata.obs.shape[0], 0))),
                        obs=adata.obs.copy(),
                    )
                adata_min.obsm["X_scVI"] = np.load(os.path.join(output_dir, "scANVI_models", label_model_name, "X_scVI.npy"))
                
                try:
                    rsc.pp.neighbors(adata_min, use_rep="X_scVI")
                    rsc.tl.umap(adata_min)
                except:
                    sc.pp.neighbors(adata_min, use_rep="X_scVI")
                    sc.tl.umap(adata_min)

                for a in labels_keys:
                    adata_min.obs.loc[adata_min.obs[a] == "Unknown", a] = np.nan

                sc.pp.subsample(adata_min, fraction=1)
                sc.pl.umap(
                    adata_min,
                    color=ref_vars,
                    sort_order=False,
                    frameon=False,
                    ncols=2,
                    na_color="red",
                    legend_loc="on data",
                    size=np.min([5e5 / adata_min.shape[0], 100])
                )
                del adata_min
                
        else:
            for z,k in enumerate(adata_ref.obs[labels_keys[i - 1]].cat.categories):
                if k == "Unknown":
                    continue
                    
                print(str(datetime.now()) + " --- Training model predicting " + j + " on " + labels_keys[i - 1] + "_scANVI=" + k + " subsetted dataset")
                model_name = hashlib.md5(str(json.dumps({**{labels_keys[i - 1]: k}, **get_model_genes_kwargs, **run_scVI_kwargs})).replace("/", " ").encode()).hexdigest()
                label_model_name = hashlib.md5(str(json.dumps({**{labels_keys[i - 1]: k}, **get_model_genes_kwargs, **run_scANVI_kwargs})).replace("/", " ").encode()).hexdigest()
                
                cells = adata.obs[labels_keys[i - 1] + "_scANVI"] == k
                if any(cells) == False:
                    continue
                    
                adata.obs[labels_keys[i - 1]] = adata.obs[labels_keys[i - 1]].astype("object")
                adata.obs[labels_keys[i - 1] + "_scANVI"] = adata.obs[labels_keys[i - 1] + "_scANVI"].astype("object")
                adata.obs.loc[(adata.obs[labels_keys[i - 1]] != adata.obs[labels_keys[i - 1] + "_scANVI"]) & (cells), j] = "Unknown"
                adata.obs[labels_keys[i - 1]] = adata.obs[labels_keys[i - 1]].astype("category")
                adata.obs[labels_keys[i - 1] + "_scANVI"] = adata.obs[labels_keys[i - 1] + "_scANVI"].astype("category")
                
                if os.path.exists(os.path.join(output_dir, "scANVI_models", label_model_name)) == False:

                    if os.path.exists(os.path.join(output_dir, "scVI_models", model_name)) == False:
                        ref_cells = adata_ref.obs[labels_keys[i - 1]] == k

                        if user_genes == None:
                            markers = get_model_genes(adata_ref[ref_cells], **get_model_genes_kwargs)
                        else:
                            if isinstance(user_genes[i], dict):
                                markers = user_genes[i][k]
                            else:
                                markers = user_genes[i]

                        model = run_scVI(adata[cells, markers], **run_scVI_kwargs)
                        model.save(os.path.join(output_dir, "scVI_models", model_name))
                        pd.DataFrame(markers).to_csv(
                            os.path.join(output_dir, "scVI_models", model_name, "var_names.csv"),
                            index=False,
                            header=False
                        )
                        if save_latent_space == True:
                            pd.DataFrame(adata.obs_names).to_csv(
                                os.path.join(output_dir, "scVI_models", model_name, "obs_names.csv"),
                                index=False,
                                header=False
                            )
                            latent_space = model.get_latent_representation()
                            np.save(
                                file=os.path.join(output_dir, "scVI_models", model_name, "X_scVI.npy"),
                                arr=latent_space
                            )
                            del latent_space

                    else:
                        markers = pd.read_csv(os.path.join(output_dir, "scVI_models", model_name, "var_names.csv"), header=None)
                        markers = markers[0].to_list()
                        model = scvi.model.SCVI.load(os.path.join(output_dir, "scVI_models", model_name), adata[cells, markers].copy())
                    
                    try:
                        label_model, probabilities = run_scANVI(adata[cells, markers], model=model, **run_scANVI_kwargs)
                        label_model.save(os.path.join(output_dir, "scANVI_models", label_model_name))
                        pd.DataFrame(markers).to_csv(
                            os.path.join(output_dir, "scANVI_models", label_model_name, "var_names.csv"),
                            index=False,
                            header=False
                        )
                        probabilities.to_csv(os.path.join(output_dir, "scANVI_models", label_model_name, "probabilities.csv"))
                    except IndexError:
                        continue

                    if save_latent_space == True or plot_latent_space == True:
                        if save_latent_space == False and plot_latent_space == True:
                            warnings.warn("The scANVI model latent space is saved by default when plot_latent_space==True and save_latent_space==False.")

                        pd.DataFrame(adata[cells, markers].obs_names).to_csv(
                            os.path.join(output_dir, "scANVI_models", label_model_name, "obs_names.csv"),
                            index=False,
                            header=False
                        )
                        latent_space = label_model.get_latent_representation()
                        np.save(
                            file=os.path.join(output_dir, "scANVI_models", label_model_name, "X_scVI.npy"),
                            arr=latent_space
                        )
                        del latent_space

                else:
                    probabilities = pd.read_csv(os.path.join(output_dir, "scANVI_models", label_model_name, "probabilities.csv"), index_col=0)
                
                probabilities = probabilities.dropna(axis=1, how='all')
                probabilities = probabilities.drop([l for l in probabilities.columns if l.startswith("_")], axis=1)
                tmp = pd.concat([adata.obs, probabilities], axis=1)
                for l in [m for m in tmp.columns if m.endswith("_y")]:
                    l = l.replace("_y", "")
                    tmp[l + "_x"] = tmp[l + "_x"].astype("object")
                    tmp[l + "_y"] = tmp[l + "_y"].astype("object")
                    tmp[l] = tmp[l + "_y"].fillna(tmp[l + "_x"])
                    tmp[l] = tmp[l].astype("category")
                    tmp = tmp.drop([l + "_y", l + "_x"], axis=1)
                adata.obs = tmp.copy()
                
                confs = [l for l in adata.obs.columns if l.endswith("_conf_scANVI")]
                for l in confs:
                    adata.obs[l] = adata.obs[l].astype('float')

                try:
                    conf_mat = adata.obs.loc[cells, :].groupby([j, j + "_scANVI"]).size().unstack(fill_value=0)
                except:
                    warnings.warn("Caught an error trying to compute the confusion matrix, printing the obs dataframe below. Does it look at expected?")
                    print(adata.obs.loc[cells, :])

                if plot_confusion == True:
                    try:
                        display(conf_mat)
                    except:
                        print(conf_mat)

                conf_mat = conf_mat.div(conf_mat.sum(axis=1), axis=0)
                conf_mat[conf_mat.isna()] = 0
                conf_mat = conf_mat.reindex(sorted(conf_mat.columns), axis=1)
                conf_mat = conf_mat.reindex(sorted(conf_mat.index), axis=0)
                
                for l in conf_mat.index:
                    if l == "Unknown":
                        continue
                    elif l not in conf_mat.columns:
                        print("WARNING: Label " + l + " fell below accruacy threshold " + str(min_accuracy) + " on reference cells. Label was not used.")
                    elif conf_mat.loc[l,l] < min_accuracy:
                        print("WARNING: Label " + l + " fell below accruacy threshold " + str(min_accuracy) + " on reference cells. Accuracy=" + str(conf_mat.loc[l,l]))

                if plot_confusion == True:
                    plt.figure(figsize=(10, 10))
                    ax = plt.pcolor(conf_mat)
                    ax = plt.xticks(np.arange(0.5, len(conf_mat.columns), 1), conf_mat.columns, rotation=90)
                    ax = plt.yticks(np.arange(0.5, len(conf_mat.index), 1), conf_mat.index)
                    plt.xlabel("Predicted")
                    plt.ylabel("Observed")
                    plt.colorbar()
                    plt.show()

                if plot_latent_space == True:
                    adata_min = ad.AnnData(
                            X=sp_sparse.csr_matrix(np.zeros((adata.obs.loc[cells, :].shape[0], 0))),
                            obs=adata.obs.loc[cells, :].copy(),
                        )
                    
                    adata_min.obsm["X_scVI"] = np.load(os.path.join(output_dir, "scANVI_models", label_model_name, "X_scVI.npy"))

                    try:
                        rsc.pp.neighbors(adata_min, use_rep="X_scVI")
                        rsc.tl.umap(adata_min)
                    except:
                        sc.pp.neighbors(adata_min, use_rep="X_scVI")
                        sc.tl.umap(adata_min)

                    for a in labels_keys:
                        adata_min.obs.loc[adata_min.obs[a] == "Unknown", a] = np.nan

                    sc.pp.subsample(adata_min, fraction=1)
                    sc.pl.umap(
                        adata_min,
                        color=ref_vars,
                        sort_order=False,
                        frameon=False,
                        ncols=2,
                        na_color="red",
                        legend_loc="on data",
                        size=np.min([5e5 / adata_min.shape[0], 100])
                    )
                    del adata_min
                
    tmp = pd.concat([adata.obs.iloc[:, int(np.where(adata.obs.columns == labels_keys[0] + "_stash")[0]):], adata.obs.iloc[:, np.where([i in labels_keys for i in adata.obs.columns])[0]]], axis=1)
    tmp.to_csv(os.path.join(output_dir, "iterative_scANVI_results." + str(datetime.date(datetime.now())) + ".csv"))

'''
Creates a list of genes that are either highly variable or differentially expressed in the given labels
Called from iterative_scANVI.

Input arguements:
adata_ref: (AnnData) object with reference labels
**kwargs: (dict) Passed to several functions, details below:
    
    use_hvg: (bool, default True) Whether to calculate and include highly variable genes from the reference dataset to scVI and scANVI
    
    use_de: (bool, default True) Whether to calculate and include differentially genes from the reference dataset to scVI and scANVI
    
    n_top_genes: (int or list, default 2000) Number of highly variable genes to pass in each iteration. Not used if use_hvg == False
    
    n_downsample_ref: (int or list, default 1000) Number of cells to downsample reference groups too in each iteration when 
    calculcating differentially expressed genes
    
    n_ref_genes: (int or list, default 500): Number of differentially expressed genes per group to pass in each iteration from the 
    reference dataset to scVI and scANVI

Outputs:
Returns list of unique markers from the procedure above
'''
                             
def get_model_genes(adata_ref, **kwargs):
    layer = kwargs["layer"]
    groupby = kwargs["groupby"]
    use_hvg = kwargs["use_hvg"]
    use_de = kwargs["use_de"]
    n_top_genes = kwargs["n_top_genes"]
    n_downsample_ref = kwargs["n_downsample_ref"]
    min_ref_cells = kwargs["min_ref_cells"]
    n_ref_genes = kwargs["n_ref_genes"]
    user_genes = kwargs["user_genes"]
    
    markers = []
    
    if "log1p" not in adata_ref.uns_keys():
        if layer == None:
            adata_ref.layers["log_normalized"] = adata_ref.X.copy()
        else:
            adata_ref.layers["log_normalized"] = adata_ref.layers[layer].copy()

        try:
            rsc.get.anndata_to_GPU(adata_ref, layer="log_normalized")
            rsc.pp.normalize_total(adata_ref, target_sum=1e4, layer="log_normalized")
            rsc.pp.log1p(adata_ref, layer="log_normalized")
            rsc.get.anndata_to_CPU(adata_ref, layer="log_normalized")
        except:
            sc.pp.normalize_total(adata_ref, target_sum=1e4, layer="log_normalized")
            sc.pp.log1p(adata_ref, layer="log_normalized")
    
    if use_hvg == True:
        try:
            try:
                rsc.get.anndata_to_GPU(adata_ref, layer=layer)
                rsc.pp.highly_variable_genes(adata_ref, flavor="seurat_v3", n_top_genes=n_top_genes, layer=layer)
                rsc.get.anndata_to_CPU(adata_ref, layer=layer)
            except:
                rsc.get.anndata_to_GPU(adata_ref, layer="log_normalized")
                rsc.pp.highly_variable_genes(adata_ref, min_mean=1, min_disp=0.5, max_mean=np.inf, layer="log_normalized")
                rsc.get.anndata_to_CPU(adata_ref, layer="log_normalized")
        except:
            try:
                sc.pp.highly_variable_genes(adata_ref, flavor="seurat_v3", n_top_genes=n_top_genes, layer=layer)
            except:
                sc.pp.highly_variable_genes(adata_ref, min_mean=1, min_disp=0.5, max_mean=np.inf, layer="log_normalized")

        markers = adata_ref.var[adata_ref.var.highly_variable == True].index.to_list()
        
    if use_de == True:
        if np.setdiff1d(adata_ref.obs[groupby].cat.categories, "Unknown").shape[0] > 1:

            ref_counts = adata_ref.obs[groupby].value_counts()
            adata_ref = adata_ref[~(adata_ref.obs[groupby].isin(ref_counts[ref_counts < min_ref_cells].index))]
                    
            cells = []
            for i in adata_ref.obs[groupby].cat.categories:
                tmp_cells = adata_ref[adata_ref.obs[groupby] == i].obs_names.to_list()
                
                if len(tmp_cells) > n_downsample_ref:
                    cells = cells + random.sample(tmp_cells, k=n_downsample_ref)

                else:
                    cells.extend(tmp_cells)

            adata_ref = adata_ref[cells]

            sc.tl.rank_genes_groups(adata_ref, method="wilcoxon", tie_correct=True, groupby=groupby, pts=True, layer="log_normalized")

            result = adata_ref.uns['rank_genes_groups']
            groups = result['names'].dtype.names
            marker_genes = {group: pd.DataFrame({key: result[key][group] for key in ['names', 'pvals_adj', 'logfoldchanges']}) for group in groups}

            for group in groups:
                marker_genes[group]['pts'] = result['pts'][group][result['names'][group]].to_list()
                marker_genes[group]['pts_rest'] = result['pts_rest'][group][result['names'][group]].to_list()
                marker_genes[group].index = marker_genes[group].names
                marker_genes[group] = marker_genes[group].drop(columns=['names'])
                tmp_genes = marker_genes[group].copy()
                tmp_genes = tmp_genes[tmp_genes.pvals_adj < 0.05]
                tmp_genes = tmp_genes.sort_values(by="logfoldchanges", axis=0, ascending=False)
                markers.extend(tmp_genes.head(n_ref_genes).index.to_list())
        else:
            warnings.warn(groupby + " contains only one label. Differentially expressed genes were NOT included in the model")
    
    return np.unique(markers)

'''
Wrapper for scVI
Called from iterative_scANVI.

Input arguements:
adata: (AnnData) Merged AnnData object
**kwargs: (dict) Passed to several functions, details below:

    layer: (None or str, default None) None if unnormalized counts are in AnnData.X, else a str where they are stored in AnnData.layers
    
    categorical_covariate_keys: (list) List of categorical covariates to pass to scVI and scANVI (e.g. ["donor_name"])
    
    continuous_covariate_keys: (list) List of continuous covariates to pass to scVI and scANVI (e.g. ["n_genes"])
    
    max_epochs_scVI: (int, default 200) Number of epochs to train scVI
    
    scVI_model_args: (None or dict) kwargs passed to scvi.model.SCVI


Outputs:
Returns trained scVI model
'''

def run_scVI(adata, **kwargs):
    layer = kwargs["layer"]
    max_epochs_scVI = kwargs["max_epochs_scVI"]
    batch_key = kwargs["batch_key"]
    categorical_covariate_keys = kwargs["categorical_covariate_keys"]
    continuous_covariate_keys = kwargs["continuous_covariate_keys"]
    scVI_model_args = kwargs["scVI_model_args"]
        
    adata = adata.copy()
        
    scvi.model.SCVI.setup_anndata(
        adata,
        layer=layer,
        batch_key=batch_key,
        categorical_covariate_keys=categorical_covariate_keys,
        continuous_covariate_keys=continuous_covariate_keys
    )
    model = scvi.model.SCVI(adata, **scVI_model_args)
    model.train(max_epochs=max_epochs_scVI, early_stopping=True)
    
    return model

'''
Wrapper for scANVI
Called from iterative_scANVI.

Input arguements:
adata: (AnnData) Merged AnnData object
**kwargs: (dict) Passed to several functions, details below:

    layer: (None or str, default None) None if unnormalized counts are in AnnData.X, else a str where they are stored in AnnData.layers
    
    categorical_covariate_keys: (list) List of categorical covariates to pass to scVI and scANVI (e.g. ["donor_name"])
    
    continuous_covariate_keys: (list) List of continuous covariates to pass to scVI and scANVI (e.g. ["n_genes"])
    
    max_epochs_scANVI: (int, default 20) Number of epochs to train scANVI
    
    scANVI_model_args: (None or dict) kwargs passed to scvi.model.SCANVI.from_scvi_model


Outputs:
Returns tupple with trained scANVI model and label predictions/probabilities (pd.DataFrame)
'''
                              
def run_scANVI(adata, model, **kwargs):
    layer = kwargs["layer"]
    max_epochs_scANVI = kwargs["max_epochs_scANVI"]
    batch_key = kwargs["batch_key"]
    categorical_covariate_keys = kwargs["categorical_covariate_keys"]
    continuous_covariate_keys = kwargs["continuous_covariate_keys"]
    labels_key = kwargs["labels_key"]
    scANVI_model_args = kwargs["scANVI_model_args"]
        
    adata = adata.copy()
    
    scvi.model.SCANVI.setup_anndata(
        adata,
        layer=layer,
        batch_key=batch_key,
        categorical_covariate_keys=categorical_covariate_keys,
        continuous_covariate_keys=continuous_covariate_keys,
        labels_key=labels_key,
        unlabeled_category="Unknown"
    )
    label_model = scvi.model.SCANVI.from_scvi_model(
        model,
        unlabeled_category="Unknown",
        labels_key=labels_key,
        adata=adata,
        **scANVI_model_args
    )
    label_model.train(max_epochs=max_epochs_scANVI, early_stopping=True)
    
    adata.obs[labels_key + "_scANVI"] = label_model.predict()
    adata.obs[labels_key + "_scANVI"] = adata.obs[labels_key + "_scANVI"].astype("category")

    probabilities = label_model.predict(soft=True)

    tmp = pd.concat([adata.obs, probabilities], axis=1)
    for l in [m for m in tmp.columns if m.endswith("_y")]:
        l = l.replace("_y", "")
        tmp[l + "_x"] = tmp[l + "_x"].astype("object")
        tmp[l + "_y"] = tmp[l + "_y"].astype("object")
        tmp[l] = tmp[l + "_y"].fillna(tmp[l + "_x"])
        tmp[l] = tmp[l].astype("category")
        tmp = tmp.drop([l + "_y", l + "_x"], axis=1)

    adata.obs = tmp.copy()

    adata.obs[labels_key + "_conf_scANVI"] = 0
    adata.obs[labels_key + "_conf_scANVI"] = adata.obs[labels_key + "_conf_scANVI"].astype("float")
    adata.obs = adata.obs.copy()

    for i in adata.obs[labels_key + "_scANVI"].cat.categories:
        adata.obs[i] = adata.obs[i].astype("float")
        adata.obs.loc[adata.obs[labels_key + "_scANVI"] == i, labels_key + "_conf_scANVI"] = adata.obs[adata.obs[labels_key + "_scANVI"] == i][i]
    
    to_pass = [labels_key + "_scANVI", labels_key + "_conf_scANVI"]
    to_pass.extend(probabilities.columns)
    probabilities = adata.obs.loc[:, to_pass]
    
    return (label_model, probabilities)

'''
Writes AnnData objects to disk that contain scANVI results and UMAP projections based on the latent representation.

Input arguements:
adata_query: (AnnData) Object with unknown cells

adata_ref: (AnnData) Object with reference labels

split_key: (str) scANVI metadata value to iteratively subset and split on (e.g. subclass_scANVI)

groupby: (str) Label predicted within the split_key (e.g. cluster if split_key is subclass_scANVI)

output_dir: (str) Location to write AnnData object

date: (str) Datestamp on the iterative_scANVI results file in the output_dir

model_args: (dict): Changes to made to scVI_model_args during training (e.g. {"n_top_genes": 5000})

**kwargs: (dict) Passed to several functions, details below:

    n_cores: (int, default 1) Number of CPU cores to use when constructing the nearest neighbor graph
    
    normalize_data: (bool, default False) Whether to log-normalize AnnData.X
    
    calculate_umap: (bool, default True) Whether to project cells into 2D with UMAP


Outputs:
Writes AnnData objects to disk at output_dir/<split_key value>_scANVI.<date>.h5ad

Example Usage:
save_anndata_kwargs = {
    **{"n_cores": 32}
}

save_anndata(
    adata=adata,
    adata_ref=adata_ref,
    split_key="subclass_scANVI",
    groupby="cluster",
    output_dir=os.path.join("output_scANVI"),
    results_file="iterative_scANVI_results.2022-02-14.csv",
    model_args={"n_top_genes": 5000},
    **save_anndata_kwargs
)
'''

def save_anndata(adata_query, adata_ref, split_key, groupby, output_dir, date, diagnostic_plots=None, model_args={}, **kwargs):
        
    default_kwargs = {
        "n_cores": 1,
        "normalize_data": False,
        "calculate_umap": True,
    }
    
    kwargs = {**default_kwargs, **kwargs}
    
    n_cores = kwargs["n_cores"]
    normalize_data = kwargs["normalize_data"]
    calculate_umap = kwargs["calculate_umap"]
    
    results_file = "iterative_scANVI_results." + date + ".csv"
    
    if isinstance(adata_query, ad.AnnData) == False or isinstance(adata_ref, ad.AnnData) == False:
        raise TypeError("One or more of the AnnData objects have an incorrect type,")
        
    if adata_query.shape[1] != adata_ref.shape[1] or all(adata_query.var_names != adata_ref.var_names) != True:
        common_labels = np.intersect1d(adata_query.var_names, adata_ref.var_names)
        
        if len(common_labels) == 0:
            raise IndexError("Reference and query AnnData objects have different shapes and no overlapping var_names.")
        
        adata_ref = adata_ref[:, common_labels].copy()
        adata_query = adata_query[:, common_labels].copy()
        warnings.warn("Reference and query AnnData objects have different shapes, using " + str(len(common_labels)) + " common var_names. This may have a deterimental effect on model performance.")

    if split_key != None:
        if split_key in adata_ref.obs.columns == False or split_key in adata_query.obs.columns == False:
            raise KeyError("One or more labels_keys do not exist in the reference AnnData object.")
        
    if os.path.exists(os.path.join(output_dir, results_file)) == False:
        raise ValueError("Output directory lacks an iterative scANVI results file.")

    adata = ad.concat([adata_query, adata_ref], join="outer", merge="unique")
    del adata_query
    
    try:
        scANVI_results = pd.read_csv(os.path.join(output_dir, results_file), index_col=0)
        scANVI_results.index = [str(l) for l in scANVI_results.index]

        if scANVI_results.shape[0] != adata.shape[0]:
            common_cells = np.intersect1d(adata.obs_names, scANVI_results.index)
            adata = adata[common_cells].copy()
            print("WARNING: Mismatch between cells in scANVI results and merged AnnData object, using " + str(len(common_cells)) + " common cells. Was this expected?") 
            
        if groupby in adata.obs.columns:
            adata.obs = adata.obs.drop([groupby], axis=1)
        
        scANVI_results = scANVI_results.loc[:, np.setdiff1d(scANVI_results.columns, adata.obs.columns)]
        
        adata.obs = pd.concat([adata.obs, scANVI_results.loc[adata.obs_names, :]], axis=1)
        
    except:
        warnings.warn("WARNING: Error merging scANVI results, saving AnnData without them.")
        pass
        
    if os.path.exists(os.path.join(output_dir, "objects")) == False:
        os.makedirs(os.path.join(output_dir, "objects"))
        
    for j in adata.obs.columns:
        if adata.obs[j].dtype == bool:
            print("Correcting bool for " + j)
            adata.obs[j] = adata.obs[j].astype("object")
            adata.obs[j] = adata.obs[j].replace({True: "True", False: "False"})

    for i in adata.obs.columns[adata.obs.isna().sum(axis=0) > 0]:
        if any(adata.obs[i].notna()) == False:
            print("Dropping no-value column " + i)
            adata.obs = adata.obs.drop([i], axis=1)
                
        else:            
            replace_with = ""
            
            if isinstance(adata.obs.loc[adata.obs[i].notna(), i][0], np.float64) == True or isinstance(adata.obs.loc[adata.obs[i].notna(), i][0], np.float32) == True:
                replace_with = 0.0

            if isinstance(adata.obs[i].dtype, pd.core.dtypes.dtypes.CategoricalDtype) == True:
                adata.obs[i] = adata.obs[i].astype("object")
            
            print("Replacing NaNs with " + str(replace_with) + " for " + i + " with dtype " + str(type(adata.obs.loc[adata.obs[i].notna(), i][0])))
            adata.obs.loc[adata.obs[i].isna(), i] = replace_with
            
            if isinstance(adata.obs.loc[(adata.obs[i].notna()) & (adata.obs[i] != ""), i][0], bool) == True:
                adata.obs[i] = [str(l) for l in adata.obs[i]]
            
    if split_key != None:
            
        if isinstance(adata.obs[split_key].dtype, pd.core.dtypes.dtypes.CategoricalDtype) == False:
            adata.obs[split_key] = adata.obs[split_key].astype("category")

        if len(adata.obs[split_key].cat.categories) == 1:
            print("WARNING: Chosen split key has only 1 category.")
            
        splits = adata.obs[split_key].cat.categories
        model_split_key = split_key.replace("_scANVI", "")
        
    else:
        splits = ["All"]
        model_split_key = None

    for i in splits:            
        if i == "" or os.path.exists(os.path.join(output_dir, "objects", i.replace("/", " ") + "_scANVI." + date + ".h5ad")) == True:
            continue
            
        elif i != "All":
            cells = adata.obs[split_key] == i
            sub = adata[cells].copy()
        
        else:
            cells = adata.obs_names
            sub = adata
                    
        with parallel_backend('threading', n_jobs=n_cores):
            if normalize_data == True:
                try:
                    rsc.get.anndata_to_GPU(sub)
                    rsc.pp.normalize_total(sub, 1e4)
                    rsc.pp.log1p(sub)
                    rsc.get.anndata_to_CPU(sub)
                except:
                    sc.pp.normalize_total(sub, 1e4)
                    sc.pp.log1p(sub)

            if calculate_umap == True:
                model_name, label_model_name = get_model_names(model_split_key, i, groupby, **model_args)
                
                if os.path.exists(os.path.join(output_dir, "scANVI_models", label_model_name)) == False:
                    # To do, implement option to perform 1-off training of an scVI model
                    print("WARNING: Cannot find the scVI model, did you run iterative_scANVI?")
                
                else:
                    markers = pd.read_csv(os.path.join(output_dir, "scANVI_models", label_model_name, "var_names.csv"), header=None)
                    markers = markers[0].to_list()
                    sub_markers_only = sub[:, markers].copy()
                    
                    try:
                        sub_markers_only.obs[groupby] = sub_markers_only.obs[groupby].astype("object")
                        sub_markers_only.obs[split_key] = sub_markers_only.obs[split_key].astype("object")
                        sub_markers_only.obs[split_key.replace("_scANVI", "")] = sub_markers_only.obs[split_key.replace("_scANVI", "")].astype("object")
                        sub_markers_only.obs.loc[(sub_markers_only.obs[split_key.replace("_scANVI", "")] != sub_markers_only.obs[split_key]), groupby] = "Unknown"
                        sub_markers_only.obs.loc[sub_markers_only.obs[groupby] == "", groupby] = "Unknown"
                        sub_markers_only.obs[groupby] = sub_markers_only.obs[groupby].astype("category")
                    except KeyError:
                        pass
                    
                    scvi.model.SCANVI.setup_anndata(
                        sub_markers_only,
                        layer=model_args["layer"],
                        batch_key=model_args["batch_key"],
                        categorical_covariate_keys=model_args["categorical_covariate_keys"],
                        continuous_covariate_keys=model_args["continuous_covariate_keys"],
                        labels_key=groupby
                    )

                    label_model = scvi.model.SCANVI.load(os.path.join(output_dir, "scANVI_models", label_model_name), sub_markers_only)
                    sub.obsm["X_scVI"] = label_model.get_latent_representation()
                    try:
                        rsc.pp.neighbors(sub, use_rep="X_scVI")
                        rsc.tl.umap(sub)
                    except:
                        sc.pp.neighbors(sub, use_rep="X_scVI")
                        sc.tl.umap(sub)

            
        sub.write(os.path.join(output_dir, "objects", i.replace("/", " ") + "_scANVI." + date + ".h5ad"))

'''
Gets the scVI and scANVI model name based on args passed.
Called by save_anndata.

Input arguements:
split_key: (str) scANVI metadata value to iteratively subset and split on (e.g. subclass_scANVI)

split_value: (str) Specific value for the split_key (e.g. Astro)

groupby: (str) Label predicted within the split_key (e.g. cluster if split_key is subclass_scANVI)

**kwargs: (dict) Passed to construct model_name and label_model_name

    layer: (None or str, default None) None if unnormalized counts are in AnnData.X, else a str where they are stored in AnnData.layers
    
    categorical_covariate_keys: (list) List of categorical covariates to pass to scVI and scANVI (e.g. ["donor_name"])
    
    continuous_covariate_keys: (list) List of continuous covariates to pass to scVI and scANVI (e.g. ["n_genes"])
    
    use_hvg: (bool, default True) Whether to calculate and include highly variable genes from the reference dataset to scVI and scANVI
    
    use_de: (bool, default True) Whether to calculate and include differentially genes from the reference dataset to scVI and scANVI
    
    n_top_genes: (int or list, default 2000) Number of highly variable genes to pass in each iteration. Not used if use_hvg == False
    
    n_downsample_ref: (int or list, default 1000) Number of cells to downsample reference groups too in each iteration when 
    calculcating differentially expressed genes
    
    n_ref_genes: (int or list, default 500): Number of differentially expressed genes per group to pass in each iteration from the 
    reference dataset to scVI and scANVI
    
    max_epochs_scVI: (int or list, default 200) Number of epochs to train scVI for in each iteration
    
    max_epochs_scANVI: (int or list, default 20) Number of epochs to train scANVI for in each iteration 
    
    scVI_model_args: (None or dict, default {"n_layer": 2}) kwargs passed to scvi.model.SCVI
    
    scANVI_model_args: (None or dict) kwargs passed to scvi.model.SCANVI.from_scvi_model


Outputs:
Returns tupple with scVI and scANVI model names (str)
'''

def get_model_names(split_key, split_value, groupby, **kwargs):
    
    default_kwargs = {
        "layer": None,
        "batch_key": None,
        "categorical_covariate_keys": None,
        "continuous_covariate_keys": None,
        "use_hvg": True,
        "use_de": True,
        "n_top_genes": 2000,
        "n_downsample_ref": 1000,
        "min_ref_cells": 15,
        "n_ref_genes": 500,
        "max_epochs_scVI": 200,
        "max_epochs_scANVI": 20,
        "scVI_model_args": {"n_layers": 2},
        "scANVI_model_args": {"n_layers": 2},
        "user_genes": None
    }
        
    kwargs = {**default_kwargs, **kwargs}
    
    layer = kwargs["layer"]
    batch_key = kwargs["batch_key"]
    categorical_covariate_keys = kwargs["categorical_covariate_keys"]
    continuous_covariate_keys = kwargs["continuous_covariate_keys"]
    use_hvg = kwargs["use_hvg"]
    use_de = kwargs["use_de"]
    n_top_genes = kwargs["n_top_genes"]
    n_downsample_ref = kwargs["n_downsample_ref"]
    min_ref_cells = kwargs["min_ref_cells"]
    n_ref_genes = kwargs["n_ref_genes"]
    max_epochs_scVI = kwargs["max_epochs_scVI"]
    max_epochs_scANVI = kwargs["max_epochs_scANVI"]
    scVI_model_args = kwargs["scVI_model_args"]
    scANVI_model_args = kwargs["scANVI_model_args"]
    user_genes = kwargs["user_genes"]
    
    get_model_genes_kwargs = {
        "layer": layer,
        "groupby": groupby,
        "use_hvg": use_hvg,
        "use_de": use_de,
        "n_top_genes": n_top_genes,
        "n_downsample_ref": n_downsample_ref,
        "min_ref_cells": min_ref_cells,
        "n_ref_genes": n_ref_genes,
        "user_genes": user_genes
    }
    run_scVI_kwargs = {
        "layer": layer,
        "max_epochs_scVI": max_epochs_scVI,
        "batch_key": batch_key,
        "categorical_covariate_keys": categorical_covariate_keys,
        "continuous_covariate_keys": continuous_covariate_keys,
        "scVI_model_args": scVI_model_args,
    }
    run_scANVI_kwargs = {
        "layer": layer,
        "max_epochs_scANVI": max_epochs_scANVI,
        "batch_key": batch_key,
        "categorical_covariate_keys": categorical_covariate_keys,
        "continuous_covariate_keys": continuous_covariate_keys,
        "labels_key": groupby,
        "scANVI_model_args": scANVI_model_args,
    }
    
    if split_key == None:
        model_name = hashlib.md5(str(json.dumps({**get_model_genes_kwargs, **run_scVI_kwargs})).replace("/", " ").encode()).hexdigest()
        label_model_name = hashlib.md5(str(json.dumps({**get_model_genes_kwargs, **run_scANVI_kwargs})).replace("/", " ").encode()).hexdigest()
        
    else:
        model_name = hashlib.md5(str(json.dumps({**{split_key: split_value}, **get_model_genes_kwargs, **run_scVI_kwargs})).replace("/", " ").encode()).hexdigest()
        label_model_name = hashlib.md5(str(json.dumps({**{split_key: split_value}, **get_model_genes_kwargs, **run_scANVI_kwargs})).replace("/", " ").encode()).hexdigest()
        
    return (model_name, label_model_name)
