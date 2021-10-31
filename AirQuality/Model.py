import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
	# Reading .csv
	air_quality = pd.read_csv(os.path.join('data', 'AirQualityUCI.csv'), delimiter=';')

	# Removing 2 last columns that does not exist. Removing last 114 lines (does not cantain data).
	# Also removes first two columns, 'Date' and 'Time'
	air_quality = air_quality.iloc[:9356, 2:-2]

	# Checking correlation between all features
	plt.figure(figsize=(20,20))
	sns.heatmap(air_quality.corr(), annot=True);
	# plt.show()

	# Parsing the attributes based on highest correlation (O3, NO2, NHMC). Dropping those that will not be used.
	x_air_quality_attributes = air_quality.drop(['CO(GT)', 'PT08.S2(NMHC)', 'PT08.S4(NO2)', 'PT08.S5(O3)'], axis=1)
	# print(x_air_quality_attributes)
	y_air_quality_attributes = air_quality.drop(['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH'], axis=1)
	# print(y_air_quality_attributes)

	# Spliting test data
	x_air_quality_training, x_air_quality_test, y_air_quality_training, y_air_quality_test = train_test_split(x_air_quality_attributes, y_air_quality_attributes, test_size = 0.3, random_state = 0)
	# print(y_air_quality_test)

	# Instantiating the regressor
	regression = LinearRegression()
	regression.fit(x_air_quality_training, y_air_quality_training)
	y_air_quality_prediction = regression.predict(x_air_quality_test)

	# Showing results
	df = pd.DataFrame(data=y_air_quality_test)
	df['PT08.S1(CO) - Predicted'] = y_air_quality_prediction
	print(df)
	# print(y_air_quality_test)
	# print(y_air_quality_prediction)
	# print(prediction)
