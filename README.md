# ANN Model
Requires Python 3.x.x+

## Usage
To run a certain config, do:
``` bash
> python main.py <config ids>
```
Examples are:
``` bash
# Run config number 4
> python main.py 3
# Run config number 2, 3 and 5
> python main.py 1 2 4
```

## Datasets
Make sure to put the data in a folder called ```resources/datasets```. The data should have the following names prefixes:

- Training data: ```training_data```
- Validation data: ```validation_data```
- Test data: ```test_data```

For more details, take a look at the [dataset repository](https://github.com/kandidat-highlights/data).

## Configuration
To edit configs, take a look at the `config.yaml` file. Please prefer making new configs instead of editing old (for academic purposes). If implementing a new model, make sure to add support for it in the `main.py` file so its configs can be automatically parsed.

## Build/Run with Docker

Build with ```docker build -t YOURTAG .```

Run with ```nvidia-docker run [-v YOURLOGDIR:/app/logs] -t -rm YOURTAG python -u ./YOURENTRYPOINT.py
