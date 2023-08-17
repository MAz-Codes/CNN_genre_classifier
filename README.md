# Audio Classification with Convolutional Neural Networks (CNN)

This repository contains a Python script that demonstrates how to use a Convolutional Neural Network (CNN) for audio classification. The model utilizes MFCC (Mel-frequency cepstral coefficients) as input features for training.

## Dependencies

To run this script, ensure you have the following libraries installed:

- `json`
- `numpy`
- `sklearn`
- `tensorflow`
- `matplotlib`

You can install the required libraries using pip:

```
pip install numpy scikit-learn tensorflow matplotlib

```

## Dataset

The script uses a dataset stored in a JSON file format. The expected structure of the JSON file is:

```json
{
    "mfcc": [...],
    "labels": [...]
}
```

- `mfcc`: Contains the Mel-frequency cepstral coefficients.
- `labels`: Contains the corresponding labels for each MFCC.

The default path to the dataset is `data.json`. Ensure that you have the dataset in the correct path or modify `DATA_PATH` in the script accordingly. I made the dataset in my other classifier repo. It is not posted here, simply because it is too big, but you can find out how to create it in the other repo.

## Functions

- `load_data(data_path)`: Loads the dataset from the provided JSON file.
- `plot_history(history)`: Plots the training and validation accuracy and loss.
- `prepare_datasets(test_size, validation_size)`: Splits the dataset into training, validation, and test sets.
- `build_model(input_shape)`: Builds and returns the CNN model.
- `predict(model, X, y)`: Predicts the label for a single sample using the trained model.
