# NB! Created in Spyder 3.6 (Anaconda)

# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ExternalPlot import ExternalPlot

# Plot code is moved to separate file
ePlot = ExternalPlot()


# *** Import and rearrange data
# Import dataset
df = pd.read_csv("TrainAndValid.csv")
columns = df.columns.tolist()




# *** Quick overview of the data ***

# Plot an overview of the raw data, 2
df.hist()
plt.suptitle('Overview of data: Histogram')

# Plot percentage of null-values per variable
ePlot.plot_percentageOfNullValues(df, columns)

# Plot histogram showing number of null-values per sample
ePlot.plot_nullValuesPerSample(df)





# *** Data Preprocessing ***

df = df.set_index('SalesID')

# Remove outliers in YearMade
ePlot.plot_yearMade_step1(df)

df.YearMade = df.YearMade.replace(1000, np.nan)

ePlot.plot_yearMade_step2(df)

df = df.loc[(df.YearMade > 1945) | (df.YearMade.isna())]
df.YearMade = pd.to_numeric(df.YearMade.replace(np.nan, round(np.mean(df.YearMade))), downcast='unsigned')

# Plot result
ePlot.plot_yearMade_step3(df)



# MachineHoursCurrentMeter: Remove extreme outliers and replace NaN with median
# Replace extreme values with the local mean of all values higher than the global mean
plt.figure()
plt.suptitle("MachineHoursCurrentMeter")
plt.subplot(2,2,1)
plt.scatter(x=df.MachineHoursCurrentMeter.index , y = df.MachineHoursCurrentMeter.values)
plt.title("MachineHours per SalesID")
plt.xlabel("Index")
plt.ylabel("Values")

plt.subplot(2,2,2)
df.MachineHoursCurrentMeter.loc[df.MachineHoursCurrentMeter!=0].hist(bins=200)
ylim1 = plt.ylim()
plt.title("Histogram of MachineHours")
plt.xlabel("Values (0h is excluded in plot)")
plt.ylabel("No. of values")

df.MachineHoursCurrentMeter.clip_upper(50000, inplace=True)
df.MachineHoursCurrentMeter.fillna(df.MachineHoursCurrentMeter.median(), inplace=True)

plt.subplot(2,2,3)
plt.scatter(x=df.MachineHoursCurrentMeter.index , y = df.MachineHoursCurrentMeter.values)
plt.ylim(ylim1)
plt.title("MachineHours per SalesID without extreme outliers")
plt.xlabel("Index")
plt.ylabel("Values")

plt.subplot(2,2,4)
df.MachineHoursCurrentMeter.loc[df.MachineHoursCurrentMeter!=0].hist(bins=200)
plt.title("Histogram of MachineHours without extreme outliers")
plt.xlabel("Values")
plt.ylabel("No. of values")
del ylim1


# BOOLEAN
# Replace NaN with "None or Unspecified", combine similar categories if necessary and encode as 1 or 0
plt.figure()
series = ["Forks", "Turbocharged", "Blade_Extension", "Pushblock", "Scarifier", "Coupler_System", "Grouser_Tracks", "Backhoe_Mounting", "Ride_Control", "Engine_Horsepower", "Pattern_Changer"]
d = {"Yes": 1, "Variable": 1, "None or Unspecified": 0, "No": 0}
for i, serie in enumerate(series):
    ePlot.plot_boolRaplceNaN(df, serie, i)
    df[serie].fillna(0, inplace=True)       # Replace NaN with 0
    df[serie].replace(d, inplace=True)   # Replace 'None or Unspecified' with 0
    df[serie].replace(to_replace=r"^(.(?<!None or Unspecified))*?$", value=1,regex=True, inplace=True) # Replace values not equal to 'None or Unspecified' with 1
del d

# Ripper: Combine 'Yes', 'Single Shank', 'Multi Shank' into one True-value
df.Ripper.fillna('None or Unspecified', inplace=True)
d = {"None or Unspecified": 0, "Yes": 1, 'Single Shank': 1, 'Multi Shank': 1}
df.Ripper.replace(d, inplace=True)
del d

# Track_Type: Create new categry for unknowns
df.Track_Type.fillna('None or Unspecified', inplace=True)


# CATEGORY
# >2 categories: Replace NaN with "None or Unspecified"
plt.figure()
series = ["Transmission", "Hydraulics", "Enclosure", "Hydraulics_Flow", "Blade_Type", "Travel_Controls", "Pad_Type", "Enclosure_Type", "Tip_Control", "Coupler"]
for i, serie in enumerate(series):
    plt.subplot(2,5,i+1)
    df[serie].value_counts().plot(kind="bar")
    plt.title(serie)
    df[serie].fillna('None or Unspecified', inplace=True)
    df[serie] = pd.Categorical(df[serie])
    plt.gcf().subplots_adjust(bottom=0.2)


# Differential_Type: Replace with Most Frequent
df.Differential_Type.fillna(df.Differential_Type.value_counts().index[0], inplace=True)
df.Differential_Type = pd.Categorical(df.Differential_Type)

# Steering_Controls
df.Steering_Controls.fillna(df.Steering_Controls.value_counts().index[0], inplace=True)
df.Steering_Controls = pd.Categorical(df.Steering_Controls)

# State: Remove a few samples with state-values "Unspecified"
df.state.replace('Unspecified', np.nan, inplace = True)
df.dropna(subset=["state"], inplace = True)
df.state = pd.Categorical(df.state)

# Track_Type: [nan, 'Steel', 'Rubber'] -> Convert nan to "TrackType unknown"
df.Track_Type.nunique(), df.Track_Type.isna().sum(), df.Track_Type.unique()
plt.title(df.Track_Type.name)
df.Track_Type.fillna("TrackType unknown", inplace=True)
df.Track_Type.value_counts().plot(kind="bar")
df.Track_Type = pd.Categorical(df.Track_Type)
plt.gcf().subplots_adjust(bottom=0.2)

# UsageBand: Replace NaN with Most Common Value
plt.figure()
plt.subplot(1,2,1)
plt.title("UsageBand: Original")
df.UsageBand.fillna("NaN").value_counts()[['Low', 'Medium', 'High', 'NaN']].plot(kind="bar")
df.UsageBand = pd.Categorical(df.UsageBand)
df.UsageBand.value_counts()[['Low', 'Medium', 'High']].plot(kind="bar")

df.UsageBand.fillna(df.UsageBand.value_counts().index[0], inplace=True)

plt.subplot(1,2,2)
plt.title("UsageBand: NaN with MostFreqVal")
df.UsageBand.value_counts()[['Low', 'Medium', 'High']].plot(kind="bar")


# Drive_System: Replace NaN with Most Common Value
ePlot.plot_DriveSystemReplaceNaNWithMostFreq_1(df)

df.Drive_System.fillna(df.Drive_System.value_counts().index[0], inplace=True)
df.Drive_System = pd.Categorical(df.Drive_System)

ePlot.plot_DriveSystemReplaceNaNWithMostFreq_2(df)


# ProductGroup
df.ProductGroup = pd.Categorical(df.ProductGroup)

# ProductGroup Desc
df.ProductGroupDesc = pd.Categorical(df.ProductGroupDesc)


# Saledate timevariable
# Construct saledateYear and saledateMonth
df.saledate = pd.to_datetime(df.saledate)

# Create dictionary for days and months
day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
month_names = ["January", "February", "March", "April", "May", "June", "July","August","September", "October", "November","December"]
dayDict = {key: day_names[key] for key in range(len(day_names))}
monthDict = {key: month_names[key] for key in range(len(month_names))}

df = df.assign(saledateDay = df.saledate.dt.weekday)
df = df.assign(saledateMonth = df.saledate.dt.month)

df.saledateDay = pd.Categorical(df.saledateDay.apply(lambda x: dayDict[x]))
df.saledateMonth = pd.Categorical(df.saledateMonth.apply(lambda x: monthDict[x-1]))

df.drop('saledate', axis=1, inplace=True)



# Convert to INT

# Tire_Size
ePlot.plot_TireSizeStr2Num_1(df)

df.Tire_Size.replace(to_replace = r'[^\d.]+', value='', regex = True, inplace=True)
df.Tire_Size.replace(to_replace = '', value=np.nan, inplace=True)
df.Tire_Size = df.Tire_Size.astype(float)
df.Tire_Size.fillna(value = df.Tire_Size.mean(), inplace=True)

ePlot.plot_TireSizeStr2Num_2(df)



# Undercarriage_Pad_Width
ePlot.UndercarriagePadWidthStr2Num_1(df)

df.Undercarriage_Pad_Width.replace(to_replace = r'[^\d.]+', value='', regex = True, inplace=True)
df.Undercarriage_Pad_Width.replace(to_replace = '', value=np.nan, inplace=True)
df.Undercarriage_Pad_Width = df.Undercarriage_Pad_Width.astype(float)
df.Undercarriage_Pad_Width.fillna(value = df.Undercarriage_Pad_Width.mean(), inplace=True)

ePlot.UndercarriagePadWidthStr2Num_2(df)

# Stick_Length
def inches(inches_str):
    try:
        feet, sep, inches = inches_str.replace("\' ", "-").replace('"', '').rpartition("-")
        if feet == '':
            return np.nan

        else:
            return 12 * int(feet) + int(inches)

    except:
        return np.nan


ePlot.plot_StickLengthStr2Inch_1(df)

df.Stick_Length = df.Stick_Length.apply(inches)
df.Stick_Length.fillna(value = df.Stick_Length.mean(), inplace=True)

ePlot.plot_StickLengthStr2Inch_2(df)

# Grouser_Type
df.Grouser_Type.fillna(0, inplace=True)
d = {'Single': 1, 'Double': 2, 'Triple': 3}
df.Grouser_Type.replace(d, inplace=True)


# Blade_Width: One of the categories is called "<12". Maybe move this to x-12 to make sure only bigger values has an effect?
df.Blade_Width.fillna("NaN").value_counts().plot(kind="bar")


# ProductSize: Convert size to INT-value
d = {'Mini': 0, 'Compact': 1, 'Small': 2, 'Medium': 3, 'Large / Medium': 4, 'Large': 5}
df.ProductSize.replace(d, inplace=True)
df.ProductSize.fillna(df.ProductSize.mean(), inplace=True)


# Remove Series with  NaN-values
df = df.loc[:,(df.isnull().sum() == 0).values]


# Manually remove categories that are considered uninteresting
df = df.drop(columns=['fiBaseModel', 'fiProductClassDesc'])
df = df.drop(columns=['fiModelDesc']) # Probably interesting, but contains 5059 different categories

# Remove ID-variables
df.drop(columns=['MachineID','ModelID', 'datasource'], inplace=True)
''' Kommentar:   MachineID ser ut til å ha negativ korrelasjon med prisen
                MachineID ser videre ut til å korrelere med datasource
                datasource har bare 6 kategorier, og kan muligens tas med
                og samtidig representere MachineID litt?
'''




# *** Encode category variables with dummy encoding ***
for index, value in df.iteritems():
    if df[index].dtype.name == 'category':
        df = pd.concat([df.drop(index, axis=1), pd.get_dummies(df[index])], axis=1)



# *** Prepare for estimator ***

# Split into independent and dependent variables
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(['SalePrice'], axis=1), df.SalePrice, test_size = 0.3)

# Feature scaling
from sklearn.preprocessing import StandardScaler
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)


# Estimator 1: Ridge Regression
print("*** Ridge regression *** ")
from sklearn import linear_model
from sklearn import metrics

ridgeReg = linear_model.Ridge()               # Create Ridge Regression object
ridgeReg.fit(X_train, y_train)                # Train the model using the training sets

# Test on training set
y_train_ridgeReg = ridgeReg.predict(X_train)    # Make predictions using the training set
print("Ridge regression test set R^2 score:\t %.2f %%" % (100*metrics.r2_score(y_train, y_train_ridgeReg)))
print("Ridge regression test set RMSE score:\t%s\n" % (np.sqrt(metrics.mean_squared_error(y_train, y_train_ridgeReg))))

# Prediction on test set
y_pred_ridgeReg = ridgeReg.predict(X_test)      # Make predictions using the testing set
print("Ridge regression validation R^2 score:\t %.2f %%" % (100*metrics.r2_score(y_test, y_pred_ridgeReg)))
print("Ridge regression validation RMSE score:\t%s\n" % (np.sqrt(metrics.mean_squared_error(y_test, y_pred_ridgeReg))))

plt.figure()
plt.bar(range(X_train.shape[1]),ridgeReg.coef_)
plt.xticks(range(X_train.shape[1]), list(df.drop(['SalePrice'], axis=1)), rotation='vertical')
plt.title("Coefficients of the linear model using Ridge regression")
plt.grid(True)


# Estimator 2: Artificial Neural Network
print("\n\n*** Artificial Neural Network (50 nodes in 2 hidden layers) ***")
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential() # Initializing the ANN
classifier.add(Dense(units = 50, kernel_initializer="uniform", activation = 'relu', input_dim = X_train.shape[1]))  # Add the input layers and the first hidden layers
classifier.add(Dense(units = 50, kernel_initializer="uniform", activation = 'relu'))                                # Add the second hidden layers
classifier.add(Dense(units = 1, kernel_initializer="uniform", activation = 'relu'))                                 # Add the output layers

classifier.compile(optimizer='rmsprop', loss='mean_absolute_error', metrics=['accuracy'])                           # Compiling the ANN
classifier.fit(X_train, y_train, epochs=4, batch_size=100)

# Test on training set
y_train_ann = classifier.predict(X_train)   # Make predictions using the training set
print("ANN test set R^2 score:\t %.2f %%" % (100*metrics.r2_score(y_train, y_train_ann)))
print("ANN test set RMSE score:\t%s\n" % (np.sqrt(metrics.mean_squared_error(y_train, y_train_ann))))

# Prediction on test set
y_pred_ann = classifier.predict(X_test)     # Make predictions using the testing set
print("ANN validation R^2 score:\t %.2f %%" % (100*metrics.r2_score(y_test, y_pred_ann)))
print("ANN validation RMSE score:\t%s\n" % (np.sqrt(metrics.mean_squared_error(y_test, y_pred_ann))))





# *** Plot results ***

# YearMade vs SalePrice
plt.figure()
df.plot(x='YearMade', y = 'SalePrice', style='o')
plt.title("Highest correlation")
plt.xlabel("YearMade"), plt.ylabel("SalePrice")

# Product size vs SalePrice
ax = df.plot(x='ProductSize', y = 'SalePrice', style='o')
plt.title("Second highest correlation")
ax.set_xticklabels(['','Mini', 'Compact', 'Small', 'Medium', 'Large / Medium', 'Large'])
plt.ylabel("SalePrice")


