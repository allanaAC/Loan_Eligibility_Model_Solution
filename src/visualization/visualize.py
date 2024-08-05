import matplotlib.pyplot as plt
import seaborn as sns
import logging

def initial_data_analysis(df):
    try:
        
        print(df.head(5))
        df['Loan_Status'].value_counts().plot.bar()
        plt.show()
        print(df['Dependents'].mode()[0])
        sns.distplot(df['LoanAmount'])
        plt.show()
    
    except Exception as e:
        logging.error(" Error in initial data analysis: {}". format(e))
    
   