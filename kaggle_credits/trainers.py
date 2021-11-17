from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from kaggle_credits.get_data import get_data
from sklearn.model_selection import train_test_split


class Trainer():
    def __init__(self,X,y) :
        self.pipeline = None
        self.X = X
        self.y = y
    def set_pipeline(self):
        pipe_2 = make_pipeline(SimpleImputer(), StandardScaler())
        pipe_1 = make_pipeline(StandardScaler())

        preprocessing = ColumnTransformer([("No_Missing", pipe_1, [
            "RevolvingUtilizationOfUnsecuredLines", "age",
            "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio",
            "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
            "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse"
        ]), ("missing", pipe_2, ["MonthlyIncome", "NumberOfDependents"])])
        preprocessing = make_pipeline(preprocessing,LogisticRegression())
        return preprocessing

    def run(self):
        self.pipe = self.set_pipeline().fit(self.X,self.y)
        return self.pipe
    def evaluate(self,X_test, y_test):
        temp = self.run()
        score = temp.score(X_test, y_test)
        return score





if __name__ == "__main__" :
    df = get_data()
    y = df['SeriousDlqin2yrs']
    X = df.drop(columns='SeriousDlqin2yrs')
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.3)
    model = Trainer(X_train, y_train)
    model.run()
    print(model.evaluate(X_test, y_test))
