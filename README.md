# iterative_scANVI

Install:
(with conda to ensure proper pytorch and jaxlib libraries are used)
```
conda env create -f environment.yml
conda activate iterative_scANVI
```

(with pip, untested)
```
pip install git+ssh://git@github.com/AllenInstitute/iterative_scANVI.git
```

Example usage:

```
from iterative_scANVI import mapping as isc_mapping

adata_ref = sc.read_h5ad("reference_adata.h5ad")
adata_query = sc.read_h5ad("query_adata.h5ad")

iterative_scANVI_kwargs = {
    "batch_key": "source",
    "categorical_covariate_keys": ["donor_name"],
}

isc_mapping.iteratively_map(
    adata_query, 
    adata_ref,
    labels_keys=["class", "subclass", "cluster"],
    output_dir=os.path.join("scANVI_output"), 
    **iterative_scANVI_kwargs
)
```
