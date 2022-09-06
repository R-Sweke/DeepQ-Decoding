### Experiments

This folder contains a template for running experiments on a cluster. Most code is borrowed from the `cluster_scripts` directory.

Some differences

- The `Environment.py` and the `Functions_Library.py` are not copied to every config but added to environment via `pip`. Make sure to install `lib` contents in your environment.
- It is possible to use MWPM now. Set `static_decoder` in `fixed_config` to `None`.