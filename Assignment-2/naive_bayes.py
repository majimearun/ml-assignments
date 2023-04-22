import numpy as np
import pandas as pd

class Distribution:
    def __init__(self,label:str,alpha:float,target:str) -> None:
        self.label = label
        self.target = target
        self.prior = None
        self.cond_prob = None
        self.alpha = alpha

    def fit(self,df:pd.DataFrame):
        filtered_df = df.loc[df[self.target]==self.label]
        self.prior = len(filtered_df)/len(df)
        self.cond_prob = dict()
        for col in filtered_df.columns:
            self.cond_prob[col]=dict()
            for unique_val in filtered_df[col].unique():
                self.cond_prob[col][unique_val]=(self.alpha+len(filtered_df[col].loc[filtered_df[col]==unique_val]))/(len(filtered_df) * self.alpha*len(filtered_df.columns))

class NaiveBayes:
    def __init__(self,target:str,df:pd.DataFrame,alpha:float=0.01):
        self.target = target
        self.df = df
        self.classes : list[Distribution] = [] 
        self.alpha = alpha
    
    def train(self):
        for distribution in self.df[self.target].unique():
            distribution_class = Distribution(label=distribution,alpha=self.alpha,target=self.target)
            distribution_class.fit(df=self.df)
            self.classes.append(distribution_class)

    def classify(self,x:pd.Series):
        filtered_df = self.df.drop(columns=[self.target])
        total_prob_list = []
        for distribution in self.classes:
            total_prob=1
            for i,col in enumerate(filtered_df.columns):
                cond_probdict = distribution.cond_prob[col]
                prob_val = cond_probdict.get(x[i])
                if(prob_val==None):
                    prob_val = 1/len(filtered_df.columns)
                total_prob *= prob_val
            total_prob_list.append(total_prob*distribution.prior)
        max_index = np.array(total_prob).argmax()
        return self.classes[max_index].label.strip()



            

                                                   

        
       
    
