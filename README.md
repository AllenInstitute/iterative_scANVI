# iterative_scANVI

Requirements:

Install scanpy and scvi-tools (recommend this be done inside a clean conda environment)
```
pip install scanpy[leiden]
pip install scvi-tools
```
Place iterative_scANVI.py in your working directory

Example usage:
```
from iterative_scANVI import *

adata_ref = sc.read_h5ad("reference_adata.h5ad")
adata_query = sc.read_h5ad("query_adata.h5ad")

iterative_scANVI_kwargs = {
    "categorical_covariate_keys": ["donor_name"]
    "continuous_covariate_keys": ["n_genes"]
    "n_top_genes": [5000, 2000, 2000]
}

iterative_scANVI(
    adata_query, 
    adata_ref,
    labels_keys=["class", "subclass", "cluster"]
    output_dir=os.path.join("scANVI_output"), 
    **iterative_scANVI_kwargs
)
```
