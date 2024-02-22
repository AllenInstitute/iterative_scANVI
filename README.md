# iterative_scANVI

Requirements:

Install scanpy and scvi-tools (recommend this be done inside a clean conda environment)
```
pip install scanpy[leiden]
pip install scvi-tools==0.14.3
conda install -c pytorch pytorch=1.10.0=py3.9_cuda11.3_cudnn8.2.0_0
```

Place iterative_scANVI.py in your working directory (detailed function usage in file)

Example usage:

```
from iterative_scANVI import *

adata_ref = sc.read_h5ad("reference_adata.h5ad")
adata_query = sc.read_h5ad("query_adata.h5ad")

iterative_scANVI_kwargs = {
    "categorical_covariate_keys": ["donor_name"],
    "continuous_covariate_keys": ["n_genes"],
    "n_top_genes": [5000, 2000, 2000]
}

iterative_scANVI(
    adata_query, 
    adata_ref,
    labels_keys=["class", "subclass", "cluster"],
    output_dir=os.path.join("scANVI_output"), 
    **iterative_scANVI_kwargs
)
```
