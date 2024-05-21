import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")


def read_dataframe(filename):
	if filename.endswith('.csv'):
		df = pd.read_csv(filename)

		df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
		df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)
	elif filename.endswith('.parquet'):
		df = pd.read_parquet(filename)

	df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
	df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

	df = df[(df.duration >= 1) & (df.duration <= 60)]

	categorical = ['PULocationID', 'DOLocationID']
	df[categorical] = df[categorical].astype(str)
	
	return df

def q1_downloading_the_data(dataframe):

	df = pd.read_parquet(dataframe)
	len_columns = len(df.columns)

	print(f"Q1 -> Download the data for January and February 2023 \n \
	Read the data for January. How many columns are there? \n \n \
	Answer Q1: {len_columns} columns \n")

def q2_computing_duration(dataframe):

	df = pd.read_parquet(dataframe)

	df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
	df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

	standard = df.duration.std()

	print(f"Q2 -> Now let's compute the duration variable. \n \
	It should contain the duration of a ride in minutes. \n \
	What's the standard deviation of the trips duration in January? \n \n \
	Answer Q2: {standard} standard diviation \n")

def q3_dropping_outliers(dataframe):

	df = pd.read_parquet(dataframe)

	df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
	df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

	df_cutout = df[(df.duration >= 1) & (df.duration <= 60)]
	dropping_outliers=(len(df_cutout) / len(df)) * 100 

	print(f"Q3 -> Next, we need to check the distribution of the duration \n \
	variable. There are some outliers. Let's remove them and keep only \n \
	the records where the duration was between 1 and 60 minutes (inclusive). \n \
	What fraction of the records left after you dropped the outliers? \n \n \
	Answer Q3: {dropping_outliers} fraction of the records \n")

def q4_one_hot_encoding(dataframe):

	df = pd.read_parquet(dataframe)

	df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
	df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

	df_cutout = df[(df.duration >= 1) & (df.duration <= 60)]

	categorical = ['PULocationID', 'DOLocationID']
	df_cutout[categorical] = df[categorical].astype(str)

	train_dicts = df_cutout[categorical].to_dict(orient='records')

	dv = DictVectorizer()
	X_train = dv.fit_transform(train_dicts)
	one_hot = X_train.shape

	print(f"Q4 -> Let's apply one-hot encoding to the pickup and dropoff \n \
	location IDs. We'll use only these two features for our model. \n \
	- Turn the dataframe into a list of dictionaries \n \
		(remember to re-cast the ids to strings - otherwise it will label encode them) \n \
	- Fit a dictionary vectorizer \n \
	- Get a feature matrix from it \n \
	What's the dimensionality of this matrix (number of columns)? \n \n \
	Answer Q4: {one_hot} dimensionality of this matrix \n")

def q5_training_a_model(dataframe):

	df = pd.read_parquet(dataframe)

	df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
	df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

	df_cutout = df[(df.duration >= 1) & (df.duration <= 60)]

	categorical = ['PULocationID', 'DOLocationID']
	df_cutout[categorical] = df[categorical].astype(str)

	train_dicts = df_cutout[categorical].to_dict(orient='records')

	dv = DictVectorizer()
	X_train = dv.fit_transform(train_dicts)

	target = 'duration'
	y_train = df_cutout[target].values

	lr = LinearRegression()
	lr.fit(X_train, y_train)
	
	y_pred = lr.predict(X_train)
	mse = mean_squared_error(y_train, y_pred, squared=False)

	print(f"Q5 -> Now let's use the feature matrix from the previous step \n \
	to train a model. \n \
	* Train a plain linear regression model with default parameters \n \
	* Calculate the RMSE of the model on the training data \n \
	What's the RMSE on train? \n \n \
	Answer Q5: {mse} dimensionality of this matrix \n")

def q6_training_a_model():

	df = read_dataframe('./data/yellow_tripdata_2023-01.parquet')
	df_val = read_dataframe('./data/yellow_tripdata_2023-02.parquet')

	categorical = ['PULocationID', 'DOLocationID']
	target = 'duration'

	train_dicts = df[categorical].to_dict(orient='records')
	val_dicts = df_val[categorical].to_dict(orient='records')

	dv = DictVectorizer()
	X_train = dv.fit_transform(train_dicts)
	y_train = df[target].values

	X_val = dv.transform(val_dicts)
	y_val = df_val[target].values

	lr = LinearRegression()
	lr.fit(X_train, y_train)
	
	y_pred = lr.predict(X_val)
	mse = mean_squared_error(y_val, y_pred, squared=False)

	print(f"Q6 -> Now let's apply this model to the validation dataset \n \
	(February 2023). \n \
	What's the RMSE on validation? \n \n \
	Answer Q6: {mse} RMSE on validation \n")

if __name__ == '__main__':

	q1_downloading_the_data('./data/yellow_tripdata_2023-01.parquet')
	q2_computing_duration('./data/yellow_tripdata_2023-01.parquet')
	q3_dropping_outliers('./data/yellow_tripdata_2023-01.parquet')
	q4_one_hot_encoding('./data/yellow_tripdata_2023-01.parquet')
	q5_training_a_model('./data/yellow_tripdata_2023-01.parquet')
	q6_training_a_model()