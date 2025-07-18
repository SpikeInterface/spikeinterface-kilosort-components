# Kilosort sorting components for SpikeInterface

This is a friendly port of some [Kilosort](https://github.com/MouseLand/Kilosort/) parts to be adapted in the SpikeInterface sortingcomponents framework.

At the moment only theses components are ported:
  * the template matching
  * the clustering


The initial plan was to integrate directly into SpikeInterface these lines of code, but due to Kilosort
license (GPL 3), we need an intermediate package with the same license.

Note that the template-matching and cluserting engine is not a copy/paste from the [Kilosort implementation]
(https://github.com/MouseLand/Kilosort/blob/main/kilosort/template_matching.py) and (https://github.com/MouseLand/Kilosort/blob/main/kilosort/clustering_qr.py), but rather an adaptation to the SpikeInterface framework.
Only a portion of the code is from the original file.


## Installation

Currently, this package can only be installed from source and it's used for benchmarking purposes:

```bash
git clone https://github.com/SpikeInterface/spikeinterface-kilosort-components.git
cd spikeinterface-kilosort-components
pip install -e .
```

