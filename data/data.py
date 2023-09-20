import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from django.http import HttpResponse
from django.shortcuts import render
import seaborn as sns
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers, Sequential

from django.views.decorators.csrf import csrf_exempt



class data():
    def __init__(self, csv, target, test_data=None):
        self.csv = csv
        self.target = target
        
        self.test_data = test_data
        self.highly_unique_col = None
        self.model = None
    
    def load_data(self):
        df = pd.DataFrame.from_dict(self.csv)
        return df
    
    def clean_data(self):
        df = self.load_data()
        missing_values = df.isna().sum().to_frame()
        missing_values = missing_values.rename(columns= {0: 'missing_values'})
        missing_values['% of total'] = (missing_values['missing_values'] / df.shape[0]).round(2) * 100
        
        new_df = df.fillna(method='ffill')
        
        df_head = new_df.head(30)
        
        
        
        return missing_values, df_head, new_df


    def visual_missing_values(self):
        df = self.clean_data()[0]
                 
        fig = plt.figure(figsize=(20, 20), dpi=50)
        fig.set_facecolor('#29556e')
        fig.suptitle(f"Missing Values", fontsize=30)
        ax = sns.barplot(x=df.index, y=df['missing_values'], palette='viridis')
        ax.set_facecolor("#031824")
        # Save the plot to a PNG image file
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        # Return the PNG image file as a Django HttpResponse object
        response = HttpResponse(buffer, content_type='image/png')
        response['Content-Disposition'] = 'inline; filename=plot.png'

        return response
     
    def prep_data(self):
        df = self.clean_data()[2]
        
        #
        self.highly_unique_col = []
        for col in df.columns:
            if len(df[str(col)].unique())/len(df) * 100 > 30 :
                self.highly_unique_col.append(col)
                
        df1 = df.drop(self.highly_unique_col, axis=1)
        df1 = df1.drop(str(self.target), axis=1)
        
        
        
        #select non numeriacl columns
        self.non_numerical_cols = df1.select_dtypes(include='object')\
                                .columns.tolist()
        
        #select bool columns
        self.bool_col = df1.select_dtypes(include='bool').columns.tolist()
        
        #get dummy variable
        if not self.non_numerical_cols:
            df2 = df1
        else:
            df2 = pd.get_dummies(df1, columns=self.non_numerical_cols)


        
        df3 = df2.copy()

        if not self.bool_col:
            df3 = df3.copy()
        else:
            df3[self.bool_col] = df2[self.bool_col].replace([True, False], [1, 0])
        
        
        
        # split data to x and y
        x = df3  #.drop(str(self.target), axis=1)        
        y = df[str(self.target)]
        
        #encode label with multiple class
                    
        self.le = LabelEncoder()
        self.le.fit(y)
        y_encode = self.le.transform(y)
        
        #visualise df
        visual_df = df3.copy()
        visual_df[str(self.target)] = y
        
        #split data into traing and test
        x_train, x_test, y_train, y_test = train_test_split(x, y_encode, test_size=0.3, random_state=123)
        
        #normalize data
        self.scaler = StandardScaler()
        self.scaler.fit(x_train)
        x_train_scaled = self.scaler.transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)
        
        return x_train_scaled, y_train, x_test_scaled, y_test, visual_df
    
    
    def deep_learning_model(self):
        """deep learning model
        this model contains just dense layers
        """
        EPOCHS=50
        number_of_epochs = range(0,EPOCHS)
        x_train, y_train, x_test, y_test = self.prep_data()[0:4]
        self.class_num = len(np.unique(y_train))
        
        #model for binary classification
        if self.class_num == 2:
            
            model = Sequential(layers=[
            layers.Input(shape=(x_train.shape[1],), dtype='float32'),
            layers.Dense(512, activation='relu'),
            layers.Dense(1, activation='sigmoid')
            ])
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            
            model.compile(optimizer=optimizer,\
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), 
                                 metrics='accuracy')
            
            history = model.fit(
                          x_train, y_train, epochs=EPOCHS, 
                          batch_size=4096,
                          validation_data=(x_test, y_test), 
                          verbose=0
                                            
                          )
            training_result = pd.DataFrame(history.history)
            
            
            
            
        
        else:
            
            model = Sequential(layers=[
            layers.Input(shape=(x_train.shape[1],), dtype='float32'),
            layers.Dense(512, activation='relu'),
            layers.Dense(self.class_num, activation='softmax')
            ])
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            
            model.compile(optimizer=optimizer,\
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
                                 metrics='accuracy')
            history = model.fit(
                          x_train, y_train, epochs=EPOCHS, 
                          batch_size=4096,
                          validation_data=(x_test, y_test),
                          verbose=0
                                            
                          )
            training_result = pd.DataFrame(history.history)
        
        
        
        self.model = model
            
           
            
        return training_result
    
    def prep_data_test(self):
        
        if self.highly_unique_col is None:
            self.prep_data()
            
        df = pd.DataFrame.from_dict(self.test_data)
        
        df1 = df.drop(columns=self.highly_unique_col, axis=1)



        #get dummy variable
        df2 = pd.get_dummies(df1, columns=self.non_numerical_cols)

        df3 = df2.copy()

        df3[self.bool_col] = df2[self.bool_col].replace([True, False], [1, 0])

        #normalize data

        x_test = self.scaler.transform(df3)

        return x_test, df
        
        
    
    
    def predictions(self):
        
        if self.model is None:
            self.deep_learning_model()
            
        x_test = self.prep_data_test()[0]
        
        df = self.prep_data_test()[1]
        
        model = self.model
        
        
        if self.class_num == 2:
            y_pred = (model.predict(x_test) > 0.5).astype("int32")
        
        else:
            y_pred = np.argmax((model.predict(x_test)), 1)
            
        prediction = self.le.inverse_transform(y_pred.ravel())
            
        df['prediction'] = prediction

        csv_data = df.to_csv(index=False)

        # create an HTTP response with the CSV data as a file attachment
        response = HttpResponse(csv_data, content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename=data.csv'

        
        return df.head(20), df, response
    



    def visual_prediction(self):
        df = self.predictions()[1]
        fig = plt.figure(figsize=(30, 30), dpi=200)
        fig.set_facecolor('#29556e')
        fig.suptitle(f"Visualization of predicted {self.target}", fontsize=30)

        ax = sns.countplot(x=df['prediction'], hue=df['prediction'], palette="viridis", linewidth=2.5)
        ax.set_facecolor("#031824")


        

        # Save the plot to a PNG image file
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        # Return the PNG image file as a Django HttpResponse object
        response = HttpResponse(buffer, content_type='image/png')
        response['Content-Disposition'] = 'inline; filename=plot.png'

        return response
        
            
        
        
    

            
    def visualise_data(self):
        """visualise training result"""
        
        df = self.deep_learning_model()
        
        
        column_names = df.columns.tolist()

        # create a figure with subplots for each column
        num_cols = len(column_names)
        num_rows = 1
        if num_cols > 1:
            num_rows = int(num_cols / 2) + (num_cols % 2 > 0)
        fig, axes = plt.subplots(num_rows, 2, figsize=(10, 5*num_rows))
        fig.set_facecolor('#29556e')
        fig.suptitle(f"Training Result with target class {self.target}", fontsize=30)
        

        # plot each column in a separate subplot
        for i, col_name in enumerate(column_names):
            row_index = int(i / 2)
            col_index = i % 2
            ax = axes[row_index, col_index] if num_rows > 1 else axes[col_index]
            ax.set_facecolor('#29556e')
            ax.plot(df[col_name])
            ax.set_title(col_name)
            ax.set_xlabel('Number of epochs')
            ax.set_ylabel(str(col_name))

        # adjust the layout and display the plot
        #plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        # Return the PNG image file as a Django HttpResponse object
        response = HttpResponse(buffer, content_type='image/png')
        response['Content-Disposition'] = 'inline; filename=plot.png'

        return response
            
        
        


    def plot_columns(self):
        df = self.prep_data()[-1]

        
        columns = df.columns.tolist()
        column_names = []
        for col in columns:
            num = df[col].nunique()
            if  num < 20:
                column_names.append(col)


        fig_width = 20
        fig_height = len(column_names) * 3
        fig = plt.figure(figsize=(fig_width, fig_height), )
        #fig = plt.figure(figsize=(30, 30), dpi=200)
        fig.set_facecolor('#29556e')
        fig.suptitle(f"Visualization of other independent Variables with {self.target}", fontsize=30)
        #fig.subplots_adjust(top=0.1)
        grid = plt.GridSpec(len(column_names), 3, wspace=0.4, hspace=0.6)


        for i, column_name in enumerate(column_names):
            ax = fig.add_subplot(grid[i//3, i%3])
            sns.countplot(x=df[column_name], hue=df[str(self.target)], palette="viridis", linewidth=2.5)
            ax.set_facecolor("#031824")
            plt.title(column_name)

            ax.get_legend().remove()

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left')
        


        

        # Save the plot to a PNG image file
        buffer = io.BytesIO()
        plt.savefig(buffer, bbox_inches='tight', format='png')
        buffer.seek(0)

        # Return the PNG image file as a Django HttpResponse object
        response = HttpResponse(buffer, content_type='image/png')
        response['Content-Disposition'] = 'inline; filename=plot.png'

        return response
    