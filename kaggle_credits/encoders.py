from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline




pipe_2 = make_pipeline(SimpleImputer(), StandardScaler())
pipe_1 = make_pipeline(StandardScaler())

preprocessing = ColumnTransformer([("No_Missing", pipe_1, [
    "RevolvingUtilizationOfUnsecuredLines", "age",
    "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio",
    "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse"
]), ("missing", pipe_2, ["MonthlyIncome", "NumberOfDependents"])])

preprocessing
