# Requirements:
 - Python 3
 - scikit-learn>=0.22.1

# Usage

```
python3 -m main <Unlabeled set (U)> <output file> <alpha (ignored)> <budget> <ndim> <distance threshold> [n_neighbors]
```

where:

- Unlabeled set is contains the inut vectors in a libsvm format
- ndim is the number of dimensions of the input vectors
- distance_threshold is a tuning parameter that defines the maximum distance to consider two instances redundant
- n_neighbors (default 100) is the amount of neighbors to consider when calculating pairwise distances between instances.


# Example of execution:
```
python3 -m main input_example chosen_ids 0 50 25000 0.8
```

# References

This algorithm is an adaptation of 

T. Wang, X. Zhao, Q. Lv, B. Hu and D. Sun, "Density Weighted Diversity Based Query Strategy for Active Learning," 2021 IEEE 24th International Conference on Computer Supported Cooperative Work in Design (CSCWD), Dalian, China, 2021, pp. 156-161, doi: 10.1109/CSCWD49262.2021.9437695.




