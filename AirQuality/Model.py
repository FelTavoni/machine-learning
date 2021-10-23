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

	# Dropping Date and Hours
	air_quality.drop(['Date', 'Time'], axis=1, inplace=True)
	print(air_quality.head(8))

	# Checking correlation between all attributes
	plt.figure(figsize=(20,20))
	sns.heatmap(air_quality.corr(), annot=True);
	# plt.show()

	# Parsing the attributes based on highest correlation (O3, NO2, NHMC)
	x_air_quality_attributes = air_quality.drop(['CO(GT)', 'PT08.S2(NMHC)', 'PT08.S4(NO2)', 'PT08.S5(O3)'], axis=1)
	y_air_quality_attributes = air_quality.drop(['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH'], axis=1)

	# Spliting test data
	x_air_quality_training, x_air_quality_test, y_air_quality_training, y_air_quality_test = train_test_split(x_air_quality_attributes, y_air_quality_attributes, test_size = 0.3, random_state = 0)

	# Instantiating the regressor
	regression = LinearRegression()
	regression.fit(x_air_quality_training, y_air_quality_training)

	# Showing results
	prediction = regression.predict(x_air_quality_test)
	print(x_air_quality_test)
	print(y_air_quality_test)
