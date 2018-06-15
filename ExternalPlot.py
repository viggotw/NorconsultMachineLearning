import matplotlib.pyplot as plt

class ExternalPlot:
    def __init__(self):
        self.props = {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5}
        

    def plot_percentageOfNullValues(self, df, columns):
        plt.figure()
        plt.bar(columns, 100*(df.isnull().sum()/df.shape[0]))
        plt.xticks(rotation=90);
        plt.gcf().subplots_adjust(bottom=0.25)
        plt.ylabel('Percentage')
        plt.title('Percentage of null-values')
        plt.grid(True)

    def plot_nullValuesPerSample(self, df):
        plt.figure()
        temp = df.isnull().sum(axis=1)
        plt.hist(temp, bins=list(range(0, temp.max())), rwidth = 0.9)
        plt.xlabel('Null-values for a sample')
        plt.ylabel('Amount of samples that contain this amount of null-values')
        plt.title('Overview of the amount of null-values in the samples')
        plt.grid(True)
        del temp
            
    def plot_yearMade_step1(self, df):
        plt.figure()
        plt.subplot(1,3,1)
        plt.title('YearMade\nOriginal')
        plt.plot(df.YearMade, 'ro')
        plt.grid(True)
        plt.xlabel('Sample index')
        plt.ylabel('Year')
        textstr = 'Max=%.2f\nMin=%.2f\nMean=%.2f' % (df.YearMade.max(), df.YearMade.min(), df.YearMade.mean())
        plt.text(1e6, 1800, textstr, fontsize=14, verticalalignment='top', bbox=self.props)
        
    def plot_yearMade_step2(self, df):
        plt.subplot(1,3,2)
        plt.title('Raplaced (YearMade=1000) with NaN')
        plt.plot(df.YearMade, 'ro')
        plt.grid(True)
        plt.xlabel('Sample index')
        plt.ylabel('Year')
        textstr = 'Max=%.2f\nMin=%.2f\nMean=%.2f' % (df.YearMade.max(), df.YearMade.min(), df.YearMade.mean())
        plt.text(1e6, 2000, textstr, fontsize=14, verticalalignment='top', bbox=self.props)
        
    def plot_yearMade_step3(self, df):
        plt.subplot(1,3,3)
        plt.title('Removed outliers and\nreplaced NaN with mean()')
        plt.plot(df.YearMade, 'ro')
        plt.grid(True)
        plt.xlabel('Sample index')
        plt.ylabel('Year')
        textstr = 'Max=%.2f\nMin=%.2f\nMean=%.2f' % (df.YearMade.max(), df.YearMade.min(), df.YearMade.mean())
        plt.text(1e6, 2000, textstr, fontsize=14, verticalalignment='top', bbox=self.props)
        
    def plot_boolRaplceNaN(self, df, serie, i):
        plt.subplot(3,4,i+1)
        plt.title(serie)
        df[serie].value_counts().plot(kind="bar", rot=0)
        
    def plot_DriveSystemReplaceNaNWithMostFreq_1(self, df):
        plt.figure()
        plt.subplot(1,2,1)
        plt.title("Drive_System: Original")
        df.Drive_System.value_counts().plot(kind="bar")
        plt.gcf().subplots_adjust(bottom=0.2)

    def plot_DriveSystemReplaceNaNWithMostFreq_2(self, df):
        plt.subplot(1,2,2)
        plt.title("Drive_System: NaN with MostFreqVal")
        df.Drive_System.value_counts().plot(kind="bar")
        
    def plot_TireSizeStr2Num_1(self, df):
        plt.figure()
        plt.suptitle("TireSize: Converting string to float")
        plt.subplot(1,2,1)
        plt.title("Original")
        df.Tire_Size.value_counts().plot(kind="bar")
        
    def plot_TireSizeStr2Num_2(self, df):
        plt.subplot(1,2,2)
        df.Tire_Size.hist()
        plt.title("Float values")
        
    def UndercarriagePadWidthStr2Num_1(self, df):
        plt.figure()
        plt.suptitle("UndercarriagePadWidth: Converting string to float")
        plt.subplot(1,2,1)
        plt.title("Original")
        df.Undercarriage_Pad_Width.value_counts().plot(kind="bar")
        
    def UndercarriagePadWidthStr2Num_2(self, df):
        plt.subplot(1,2,2)
        plt.title("Int values")
        df.Undercarriage_Pad_Width.hist(bins=20)
        
    def plot_StickLengthStr2Inch_1(self, df):
        plt.figure()
        plt.suptitle("StickLength: Converting string to float")
        plt.subplot(1,2,1)
        plt.title("Original")
        df.Stick_Length.value_counts().plot(kind="bar")
        
    def plot_StickLengthStr2Inch_2(self, df):
        plt.subplot(1,2,2)
        plt.title("Int values")
        df.Stick_Length.hist()