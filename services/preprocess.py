from typing import Tuple, Optional
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import io
from fastapi.responses import StreamingResponse
from fastapi import Query

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

from models.enums import PlotType

def combine_rare_categories(series:pd.Series, threshold=50):
    value_counts = series.value_counts()
    rare_categories = value_counts[value_counts < threshold].index
    return series.apply(lambda x: 'Other'if x in rare_categories else x)


def preprocess_model_clust_adj_4(df: pd.DataFrame):
    """
    Prepare the dataset for training.
    """
    rename_mapper = {
    'textbox20' : 'Adjustment_Amount_Per_Invoice_Number',
    'Svc_Date' : 'Service_Date',
    'Rev_Type' : 'Revenue_Type',
    'Trans_Date' : 'Transaction_Date',
    'Phy_Name' : 'Physician_Name',
    'Proc_Code' : 'Process_Code',
    'Adj_Amt' : 'Adjustment_Amount'
    }

    df.rename(columns=rename_mapper, inplace = True)
    df.drop(['Inv_Num', 'Process_Code', 'Crt_UserID',
          'Client_ID', 'Date_Updated', 'Adjustment_Amount_Per_Invoice_Number'],
          axis = 1, inplace = True
          )

    df['Service_Date'] = pd.to_datetime(df['Service_Date'])
    df['DayOfWeek'] = df['Service_Date'].dt.dayofweek
    df['DayOfWeekName'] = df['Service_Date'].dt.day_name()

    df.drop('Payer_Name', axis = 1, inplace = True)

    X = df[['Clinic', 'Revenue_Type', 'Physician_Name',
         'Adjustment_Amount', 'Reason', 'DayOfWeekName']]
    y = df['Payer']

    categorical_cols = ['Clinic', 'Revenue_Type', 'Physician_Name',
                        'Reason', 'DayOfWeekName'
                        ]

    rare_cols = ['Clinic', 'Revenue_Type', 'Physician_Name',
                 'Reason', 'DayOfWeekName'
                 ]

    mask = y != 'Other'

    X = X[mask].reset_index(drop = True)
    y = y[mask].reset_index(drop = True)

    for col in rare_cols:
        try:
            X[col] = combine_rare_categories(X[col], col)
        except Exception as e:
            print('An exception occurred, {e}')
            pass

    X_scaled = X.copy(deep = True)

    scaler = StandardScaler()
    X_scaled['Adjustment_Amount'] = scaler.fit_transform(X[['Adjustment_Amount']])

    encoder = OrdinalEncoder()
    X_scaled[categorical_cols] = encoder.fit_transform(X_scaled[categorical_cols])
    X_scaled.dropna(subset = 'Revenue_Type', inplace = True)

    kmeans = KMeans(n_clusters = 4, random_state = 42)
    X_scaled['Cluster'] = kmeans.fit_predict(X_scaled)

    cluster_summary = pd.DataFrame(X_scaled.groupby('Cluster').mean())
    print(cluster_summary)

    X_pca_input = X_scaled.drop('Cluster', axis = 1)

    pca = PCA(n_components = 2)
    pca_components = pca.fit_transform(X_pca_input)

    pca_df = pd.DataFrame(pca_components, columns = ['PC1', 'PC2'])
    pca_df['Cluster'] = X_scaled['Cluster'].values


    plt.figure(figsize = (8, 6))
    sns.scatterplot(data = pca_df, x = 'PC1', hue = 'Cluster', y = 'PC2', palette = 'tab10')
    plt.title('Cluster visualization using PCA')
    plt.show()

    feature_names = X_scaled.drop('Cluster', axis = 1)

    loadings = pd.DataFrame(pca.components_.T, 
                            columns=['PC1', 'PC2'], 
                            index=feature_names.columns
                            )

    X_pca = pca_components
    y = y.reset_index(drop = True)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(random_state = 42, class_weight = 'balanced')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    return{
        "classification_report" : classification_report(y_test, y_pred),
        "cluster_summary" : cluster_summary
    }


def preprocess_linear_reg(lin_df: pd.DataFrame, plot_type: Optional[PlotType] = None) -> Tuple[dict, str]:
    '''
    Preprocess the dataset for linear_regression.

    Args:
    :param lin_df: Dataframe to process
    :type lin_df: pd.DataFrame
    :param plot_type: types of plotting available. Valid types are
                        "hist", "box", "scatter", "pairplot",
                        "heatmap", "violin", "barplot", "lineplot",
                        "kde"
    :type plot_type: str
    '''
    lin_df.drop('id', axis = 1, inplace = True)
    lin_df.plot.scatter('rm', 'medv')
    plt.subplots(figsize=(12,8))
    sns.heatmap(lin_df.corr(), cmap = 'RdGy')
    sns.pairplot(lin_df, vars = ['lstat', 'ptratio', 'indus', 'tax',
                                 'crim', 'nox', 'rad', 'age', 'medv'
                                 ])
    sns.pairplot(lin_df, vars = ['rm', 'zn', 'black', 'dis', 'chas','medv'])

    X = lin_df[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age',
                'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']
                ]
    y = lin_df['medv']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    lm = LinearRegression()
    lm.fit(X_train,y_train)
    plot_path = None

    predictions = lm.predict(X_test)

    if plot_type == 'hist':
        fig, ax = plt.subplots()
        lin_df['medv'].hist(bins=30, ax = ax)
        ax.set_title('Distribution of medv')
        ax.set_xlabel('medv')
        ax.set_ylabel('Frequency')
        plt.tight_layout()
        plot_path = 'static/plots/medv_histogram.png'
        plt.savefig(plot_path, dpi=300)

    elif plot_type == 'box':
        fig, ax = plt.subplots()
        sns.boxplot(x='chas', y='medv', data=lin_df, ax = ax)
        ax.set_title('Median Home Value by Proximity to Charles River')
        ax.set_xlabel('Tract Bounds Charles River (CHAS)')
        ax.set_ylabel('Median Value of Homes ($1000s)')
        plt.tight_layout()
        plot_path = 'static/plots/medv_boxplot.png'
        plt.savefig(plot_path, dpi=300)

    elif plot_type == 'scatter':
        fig, ax = plt.subplots()
        sns.scatterplot(x='lstat', y='medv', data=lin_df, ax = ax)
        ax.set_title('Home Value vs. Percentage of Lower Status Population')
        ax.set_xlabel("% Lower Status of Population (LSTAT)")
        ax.set_ylabel("Median Value of Homes ($1000s)")
        plt.tight_layout()
        plot_path = 'static/plots/medv_scatterplot.png'
        plt.savefig(plot_path, dpi=300)

    elif plot_type == 'pairplot':
        sns.pairplot(lin_df[['rm', 'lstat', 'medv', 'tax']])
        plt.suptitle("Pairwise Relationships Between Housing Features", y=1.02)
        plt.tight_layout()
        plot_path = 'static/plots/medv_pairplot.png'
        plt.savefig(plot_path, dpi=300)

    elif plot_type == 'heatmap':
        plt.figure(figsize=(12, 10))
        sns.heatmap(lin_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", square=True, linewidths=0.5)
        plt.title("Correlation Heatmap of Boston Housing Features")
        plt.tight_layout()
        plot_path = 'static/plots/medv_heatmap.png'
        plt.savefig(plot_path, dpi=300)

    elif plot_type == 'violin':
        sns.violinplot(x='rad', y='medv', data=lin_df)
        plt.title("Distribution of Home Values by Accessibility to Radial Highways")
        plt.xlabel("Accessibility Index to Radial Highways (RAD)")
        plt.ylabel("Median Value of Homes ($1000s)")
        plt.tight_layout()
        plot_path = 'static/plots/medv_violinplot.png'
        plt.savefig(plot_path, dpi=300)

    elif plot_type == 'barplot':
        sns.barplot(x='chas', y='medv', data=lin_df)
        plt.title('Average Home Value by Charles River Proximity')
        plt.xlabel('Bounds Charles River (CHAS)')
        plt.ylabel('Average Median Home Value ($1000s)')
        plt.tight_layout()
        plot_path = 'static/plots/medv_barplot.png'
        plt.savefig(plot_path, dpi=300)

    elif plot_type == 'lineplot':
        lin_df['medv'].sort_values().reset_index(drop=True).plot()
        plt.title("Sorted Median Home Values in Boston Housing Dataset")
        plt.xlabel("Sorted Index")
        plt.ylabel("Median Home Value ($1000s)")
        plt.tight_layout()
        plot_path = 'static/plots/medv_lineplot.png'
        plt.savefig(plot_path, dpi=300)

    elif plot_type == 'kde':
        sns.kdeplot(lin_df['medv'], shade=True)
        plt.title("Density Plot of Median Home Values")
        plt.xlabel("Median Home Value ($1000s)")
        plt.ylabel("Density")
        plt.tight_layout()
        plot_path = 'static/plots/medv_kdeplot.png'
        plt.savefig(plot_path, dpi=300)

    else:
        plt.scatter(y_test,predictions)
        plt.xlabel('Y Test')
        plt.ylabel('Predicted Y')

        print('MAE:', metrics.mean_absolute_error(y_test, predictions))
        print('MSE:', metrics.mean_squared_error(y_test, predictions))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
        sns.histplot((y_test-predictions),bins=50, kde = True)
        plt.tight_layout()
        plot_path = 'static/plots/linear_reg_plot.png'
        plt.savefig(plot_path)
        plt.close()

    coefficients = pd.DataFrame(lm.coef_,X.columns)
    coefficients.columns = ['coefficients']

    print(f'Type of coefficients.to_dict: {type(coefficients.to_dict())}')
    print(f'Type of plot_path: {type(plot_path)}')

    return coefficients.to_dict(), plot_path
