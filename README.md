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

## Application Endpoint
There is a simple RESTful endpoint that can be used to make predictions on a specified title. To use the endpoint the model has to be trained alreay. Once the network is trained and ready to make some predictions, run
```
python main.py <config id> --application
```
Where `<config id>` is the ID of the config for the trained model.

The endpoint is then available at `localhost:5000/predict`. An example request is:
```
HTTP GET localhost:5000/predict?text=Hello%20World!&subreddit=funny
```
Where users for the title `Hello World!` and subreddit `funny` are predicted. It's important that the title string is URL encoded (i.e. replacing `" "` with `"%20"`). The response looks like the following:
```json
{
    "predictions": [
        "UNK",
        "izzycat",
        "binarylogik",
        "bopolissimus",
        "flezgodrit"
    ],
    "predicted": "Hello World!",
    "subreddit": "funny"
}
```
A simple client app that uses this endpoint can be found at:
[kandidat-hightlights/app](https://github.com/kandidat-highlights/app)

## Datasets
Make sure to put the data in a folder called ```resources/datasets```. The data should have the following names prefixes:

- Training data: ```training_data```
- Validation data: ```validation_data```
- Test data: ```test_data```

For more details, take a look at the [dataset repository](https://github.com/kandidat-highlights/data).

## Configuration
To edit configs, take a look at the `config.yaml` file. Please prefer making new configs instead of editing old (for academic purposes). If implementing a new model, make sure to add support for it in the `main.py` file so its configs can be automatically parsed.

## Build/Run with Docker

Build with 
```
docker build -t YOURTAG
```

Run with 
```
nvidia-docker run [-v YOURLOGDIR:/app/logs] -t -rm YOURTAG python -u ./YOURENTRYPOINT.py
```
