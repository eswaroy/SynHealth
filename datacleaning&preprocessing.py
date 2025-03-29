#comment out the required code with respect to the file you are dealing with.
import pandas as pd
admissions = pd.read_csv("ADMISSIONS(c).csv")
patients = pd.read_csv("PATIENTS(c).csv")
icustays = pd.read_csv("ICUSTAYS(c).csv")
labevents = pd.read_csv("LABEVENTS(c).csv")
chartevents=pd.read_csv("CHARTEVENTS.csv")
for df, name in zip([admissions,patients,icustays,labevents,chartevents],['ADMISSIONS','PATIENTS','ICUSTAYS','LABEVENTS','CHARTEVENTS']):
#     #print(f'null values in {name} :\n',df.isnull().sum())
#     #print(df.info())
#     # for col in df.select_dtypes(include=['object']).columns:
#     #     mode_value=df[col].mode()[0]
#     #     df[col].fillna(mode_value,inplace=True)
#     # for col in df.select_dtypes(include=['float']).columns:
#     #     df[col] = pd.to_numeric(df[col], errors='coerce')
#     #     df[col].fillna(df[col].mean(), inplace=True)
    print(f'null values in {name} :\n',df.isnull().sum())
    # print(f"Missing values in {name} after filling NaNs:\n", df.isnull().sum(),df.info())
    df.to_csv(f'{name}(c).csv')