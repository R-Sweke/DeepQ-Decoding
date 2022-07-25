### Installation of dependencies

---

As mentioned in the `README.md` this repository comes with a bunch of dependencies. Dependencies are maintained in `conda` environments. Environment files are in the `environments` folder. Currently we support two environments:
- `environment.yaml`: vanilla Tensorflow, no special CPU instruction support.
- `environment-mkl.yaml`: Intel MKL support.

Environment can be installed with `mamba` [1] or `conda` (not recommended &rarr; slow).

To install the packages run
```
mamba create env --file environment.yaml
mamba activate deepq
```

The code has also a dependency to a forked version of `keras-rl` [2]. To install it (in the activated `deepq` env) run
```
pip install git+https://github.com/R-Sweke/keras-rl
```

For convenience install the `ipykernel` so that `jupyter` finds it:
```
python -m ipykernel install --user --name deepq
```

To remove the kernel you can run
```
jupyter kernelspec uninstall deepq
```

That's it, you are ready to go!

### References
---
- [1] https://github.com/mamba-org/mamba
- [2] https://github.com/R-Sweke/keras-rl