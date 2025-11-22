# Bachelor-thesis
Interpretability of large language models for classification of protein thermostability.


## Resources
### Datasets
- [Tempura](https://togodb.org/db/tempura)
- [Uniref50](https://ftp.uniprot.org/pub/databases/uniprot/current_release/uniref/uniref50/)
- [Pfam](https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/)

## Termal class borders
- Psychrophilic: < 20 °C (< 15 °C)
- Mesophilic: 20–45 °C (30 - 35 °C)
- Thermophilic: 45–80 °C (50 - 70 °C)
- Hyperthermophilic: > 80 °C (>= 80 °C)

## Methods
### Captum
Integrated gradients ([source](https://captum.ai/docs/extension/integrated_gradients))

### Beyond Attention
Transformer Interpretability Beyond Attention Visualization ([source](https://arxiv.org/pdf/2012.09838))
