# Setting up the Anderson Dataset

1. Download the dataset from [Anderson (2016)](https://github.com/IrinaStatsLab/Awesome-CGM/wiki/Anderson-(2016))
2. Change dataset directory in `preprocess.py` and run it (any line commented with a `TODO:` must be modified)
3. The second step will create a csv file. Pass this file path into `AndersonDataLoader` class. Optionally, you can also pass the ketones file path (file in `Data Tables/Ketone.txt`). Ketones data are currently not as useful, since ketones data timestamps are completely different years from the bgl reading timestamps


**Examples in notebook `0.03` under the `Loading Anderson (2016) Example`**
