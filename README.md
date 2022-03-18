# Applied Reinforcement Learning
Applied reinforcement learning repository for MENG 25620 project
# Motivation
Protein folding is an inherently difficult computational problem, well know to be "NP-hard." This means analytical solutions are few and far between, thus motivating the need for heuristic approaches which leverage different simplifications to find an approximate solution to the free energy minimum. 

In this repository, we develop a reinforcement learning protocol which leverages two key heuristics.
* **The majority of amino acid behaivor is characterized by two properties:*** Our approach relies on the simplification that the dominating properties of protein residues comes from difference in hydrophobicity. Therefore, our model folds proteins with only two classes of amino acids: hydrophobic (+1) and hydrophilic (-1)
* ***2D Lattice Test Case:*** We further simplify the problem to that of a 2D lattice, which projects the positions of the residues from a continuous space of $\alpha$ to a discrete 2-dimensional space. Granted, this is a drastic simplification of the problem, but should serve as a good test case for the validity of the approach in general

Our work is heavily informed by 
