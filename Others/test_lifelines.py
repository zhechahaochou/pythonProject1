from datasets import load_waltons


df = load_waltons()
print(df.head(),'\n')
print(df['T'].min(), df['T'].max(),'\n')
print(df['E'].value_counts(),'\n')
print(df['group'].value_counts(),'\n')


