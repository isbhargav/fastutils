

def column_card(df: pd.DataFrame, col_type: str) -> pd.DataFrame:
    '''
    Function that returns a dataframe having fields of giveb col_type, their cardaility and unique values

    Input: Dataframe
    Output: Dataframe

    Example:
        Fields 	        Cardinlity 	Unique
    0 	UsageBand 	    4 	        [Low, High, Medium, nan]
    1 	saledate 	    3919 	    [11/16/2006 0:00, 3/26/2004 0:00, 2/26/2004 0:...
    2 	fiModelDesc     4999 	    [521D, 950FII, 226, PC120-6E, S175, 310G, 790E...
    3 	fiBaseModel 	1950 	    [521, 950, 226, PC120, S175, 310, 790, 416, 43...
    4 	fiSecondaryDesc 176 	    [D, F, nan, G, E, HAG, B, NX, SUPER K, STD, BL...
    5 	fiModelSeries 	123 	    [nan, II, -6E, LC, -5, III, -1, 5, -2, 1, #NAM...



    '''
    object_fields = [y for y, x in zip(
        df.columns.tolist(), df.dtypes) if x == col_type]

    card_objs_lens = [len(df[x].unique()) for x in object_fields]
    card_objs = [df[x].unique() for x in object_fields]

    return pd.DataFrame({'Fields': object_fields, 'Cardinlity': card_objs_lens, 'Unique': card_objs})


def object_columns(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Function that returns a dataframe having fields object, their cardaility and unique values

    Input: Dataframe
    Output: Dataframe

    Example:
        Fields 	        Cardinlity 	Unique
    0 	UsageBand 	    4 	        [Low, High, Medium, nan]
    1 	saledate 	    3919 	    [11/16/2006 0:00, 3/26/2004 0:00, 2/26/2004 0:...
    2 	fiModelDesc     4999 	    [521D, 950FII, 226, PC120-6E, S175, 310G, 790E...
    3 	fiBaseModel 	1950 	    [521, 950, 226, PC120, S175, 310, 790, 416, 43...
    4 	fiSecondaryDesc 176 	    [D, F, nan, G, E, HAG, B, NX, SUPER K, STD, BL...
    5 	fiModelSeries 	123 	    [nan, II, -6E, LC, -5, III, -1, 5, -2, 1, #NAM...



    '''

    return column_card(df, col_type='object')


class Categorify():
    "Transform the categorical variables to that type."

    def __init__(self, cat_cols: [str]):
        self.cat_names = cat_cols

    def apply_train(self, df: pd.DataFrame):
        "Transform `self.cat_names` columns in categorical."
        self.categories = {}
        for n in self.cat_names:
            df.loc[:, n] = df.loc[:, n].astype('category').cat.as_ordered()
            self.categories[n] = df[n].cat.categories

    def apply_test(self, df: pd.DataFrame):
        "Transform `self.cat_names` columns in categorical using the codes decided in `apply_train`."
        for n in self.cat_names:
            df.loc[:, n] = pd.Categorical(
                df[n], categories=self.categories[n], ordered=True)
