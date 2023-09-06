# Purchases card transactions- Predict inactivity

In this dataset you have a collection of purchase card transactions for the Birmingham City Council. https://data.birmingham.gov.uk/organization/birmingham-city-council.<br>
The data are distributed in excel spreadsheets, 1 per month from 2017 to 2023, with a few months missing.<br>

Opening 1 of the files, it can be verified that the records contain transactions per person, which will be identified by the end of the credit card, the transaction is accompanied by the value, date, and identifiers of the supplier and classifiers of the activity and type of product of this supplier.

## Installation

1. Clone the repository:

```shell
git clone https://github.com/rodrigo121998/purchase_card_trans.git
```

2. Create and activate a virtual environment (optional but recommended). You require python 3.10:
```shell
python3 -m venv venv
source venv/bin/activate
```

3. Install the required packages:
```shell
pip install -r requirements.txt
```

## Usage

- Exploratory Data Analysis: The exploratory data analysis can be found in the file `inactivity_card_profiling.html.` It provides insights into the dataset and the factors influencing inactivity of the card.

- Model Training and Benchmarking: The process of training and benchmarking the model is documented in the file `main.ipynb`. It includes the selection of the LightGBM model as the final model choice.