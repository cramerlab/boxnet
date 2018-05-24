# BoxNet

BoxNet is a deep residual convolutional network for particle picking and dirt masking in cryo-EM data. It's pre-trained on a lot of data and performs rather well.

The easiest way to use it is inside [Warp](https://github.com/cramerlab/warp). We are constantly re-training the model on the latest version of the training dataset. To benefit from that, grab the latest model from ftp://multiparticle.com/boxnet/models/. Models containing a "Mask" suffix are pre-trained to also mask out dirt in micrographs. In case this leads to bad results, try a model without masking.

The training dataset is too large to fit in this repository. It is hosted at ftp://multiparticle.com/boxnet/trainingdata/. The .zip file contains the entire corpus. To use it to re-train the model in [Warp](https://github.com/cramerlab/warp), put as many of the TIFF files as you want into the "boxnet2training" folder of your Warp installation.

To submit more data to the training repository, please use [this page](https://www.multiparticle.com/warp/?page_id=72).

**BoxNet can't pick helical filaments yet. We are especially looking for good examples of helical data to implement this feature!**

## Current list of samples

| Access code | Sample | Synthetic | Carbon support | VPP | Released |
|--- | --- | --- | --- | --- | --- |
| EMPIAR-10017 | beta-galactosidase |   |   |   | ✓ |
| EMPIAR-10077 | 80S ribosome |   | ✓ |   | ✓ |
| EMPIAR-10078 | 20S proteasome |   |   | ✓ | ✓ |
| EMPIAR-10081 | HCN1 channel |   |   |   | ✓ |
| EMPIAR-10084 | Haemoglobin |   |   | ✓ | ✓ |
| EMPIAR-10089 | TcdA1 in prepore state |   |   |   | ✓ |
| EMPIAR-10097 | Influenza Hemagglutinin |   |   |   | ✓ |
| EMPIAR-10122 | Apoferritin |   |   | ✓ | ✓ |
| EMPIAR-10153 | 80S ribosome |   | ✓ | ✓ | ✓ |
| EMPIAR-10156 | 80S ribosome |   | ✓ |   | ✓ |
| PDB-1sa0 | Tubulin-Colchicine | ✓ |   |   | ✓ |
| PDB-2gtl | Lumbricus Erythrocruorin | ✓ |   |   | ✓ |
| PDB-2wri | 70S ribosome | ✓ |   |   | ✓ |
| PDB-3j9i | 20S proteasome | ✓ |   |   | ✓ |
| PDB-4hhb | Haemoglobin | ✓ |   |   | ✓ |
| PDB-4zor | S37P MS2 viral capsid | ✓ |   |   | ✓ |
| PDB-5foj | Grapevine Fanleaf virus | ✓ |   |   | ✓ |
| PDB-5mmi | Chloroplast ribosome, large subunit | ✓ |   |   | ✓ |
| PDB-5ngm | 70S ribosome | ✓ |   |   | ✓ |
| PDB-5vy5 | Aldolase | ✓ |   |   | ✓ |
| PDB-5w3l | Rhi virus B14 | ✓ |   |   | ✓ |
| PDB-5w3s | TRPML3 channel | ✓ |   |   | ✓ |
| PDB-5xnl | Stacked PSII-LHCII supercomplex | ✓ |   |   | ✓ |
| PDB-5xwy | LbuCas13a-crRNA complex | ✓ |   |   | ✓ |
| PDB-5y6p | Phycobilisome | ✓ |   |   | ✓ |
| PDB-6az1 | 80S ribosome, small subunit | ✓ |   |   | ✓ |
| PDB-6b7n | Coronavirus spike protein | ✓ |   |   | ✓ |
| PDB-6b44 | CRISPR Csy surveillance complex | ✓ |   |   | ✓ |
| PDB-6bco | TRPM4 channel | ✓ |   |   | ✓ |
| PDB-6bcx | mTORC1 | ✓ |   |   | ✓ |
| PDB-6bhu | MRP1 | ✓ |   |   | ✓ |
| | RNA Polymerase II complex |   |   |   | ✓ |
| | RNA Polymerase II complex |   | ✓ |   | ✓ |
| | RNA Polymerase II complex |   |   |   | ✓ |
| | viral polymerase |   |   |   | ✓ |
| | nucleosome complex |   |   |   |   |



## Authorship

BoxNet is being developed by Dimitry Tegunov ([tegunov@gmail.com](mailto:tegunov@gmail.com)) in Patrick Cramer's lab at the Max Planck Institute for Biophysical Chemistry in Göttingen, Germany.
