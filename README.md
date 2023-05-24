You'll find in this repository our submission to the quantum computing [Open Science Prize 2022](https://github.com/qiskit-community/open-science-prize-2022) organized by IBM.

## Content
It consists of 3 files:
- [our_method.pdf](our_method.pdf): a document presenting our method and our decisions
- [heavy_hex_kagome_estimator.py](heavy_hex_kagome_estimator.py): a custom estimator class (as an alternative to the [Qiskit default estimator primitive](https://qiskit.org/documentation/partners/qiskit_ibm_runtime/tutorials/how-to-getting-started-with-estimator.html)) that we designed specially for [Kagome lattices](https://en.wikipedia.org/wiki/Trihexagonal_tiling#Kagome_lattice) mapped to [Heavy-Hex](https://research.ibm.com/blog/heavy-hex-lattice) quantum device architectures such as the IBMQ Gaudalupe device
- [notebook.ipynb](notebook.ipynb): a python notebook showing how to use the estimator and implement our method with qiskit

## Authors
- [Clément Boisson](https://github.com/clement-boisson)
- [Irene de León](https://github.com/IrenedeLeon)

Feel free to contact us if you have any questions!
