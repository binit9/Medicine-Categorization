from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler


class df_column_extractor(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
        
    def transform(self, df, y=None):
        return df[[self.column]]
            
    def fit(self, df, y=None):
        return self
		
		
class Converter(BaseEstimator, TransformerMixin):
    def transform(self, df, y=None):
        return df.values.ravel()
            
    def fit(self, df, y=None):
        return self