# SEMORE
## SEmi-automatic MOrphological fingerprinting and segmentation Extraction
A multi-independent-module pipeline for structure segmentation and disection in single molecule localization microscopy (SMLM) data and the extraction of unique morphological fingerprints.
![image](_Images\Figure_1.png)
### Citing
TBA
### Dependencies
  - python=3.8
  - pandas
  - matplotlib
  - scipy
  - hdbscan
  - opencv
  - scikit-learn
  - umap-learn
### Usage
#### Installation
```bash
git clone
cd SEMORE
conda env create -f environment.yml
conda activate SEMORE
```
SEMORE modules and additional function are contained in the `Scripts` folder.
SEMORE modules are imported as:
```python
from Scripts.SEMORE_clustering import find_clust
from Scripts.SEMORE_fingerprint import Morphology_fingerprint
```
Thee three test python scripts are provided:
  - `Data_sim_test.py` - test data generation.
  - `Segmentation_test.py` - test the clustering module on simulated data.
  - `Fingerprint_test.py` - test the fingerprint modules on the resultet data from Segmentation_test.py.

### Contact
https://www.hatzakislab.com/


Nikos S.Hatzakis, Professor\
Department of Chemistry\
hatzkais@chem.ku.dk

Jacob Kæstel-hansen, PhD fellow\
Department of Chemistry\
jkh@chem.ku.dk

Steen W. B. Bender, Master student\
Department of Chemistry\
csq439@alumni.ku.dk

