# iterative_scANVI

Install:
```
pip install git+ssh://git@github.com/AllenInstitute/iterative_scANVI.git
```

Example usage:

```
from iterative_scANVI import mapping as isc_mapping

adata_ref = sc.read_h5ad("reference_adata.h5ad")
adata_query = sc.read_h5ad("query_adata.h5ad")

iterative_scANVI_kwargs = {
    "categorical_covariate_keys": ["donor_name"],
    "continuous_covariate_keys": ["n_genes"],
    "n_top_genes": [5000, 2000, 2000]
}

isc_mapping.iteratively_map(
    adata_query, 
    adata_ref,
    labels_keys=["class", "subclass", "cluster"],
    output_dir=os.path.join("scANVI_output"), 
    **iterative_scANVI_kwargs
)
```
