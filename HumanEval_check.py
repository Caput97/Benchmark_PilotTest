import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd




def humanEval_check(df):
    for index, row in df.iterrows():
        if row['System_evaluation'] == row['Human_evaluation']:
            df.loc[index, 'Check'] = 'Match'
        else:
            df.loc[index, 'Check'] = 'Not_Match'


df = pd.read_csv('DF_HumanAutomaticEval.csv', index_col = 0)

humanEval_check(df)

Match_count = df['Check'].value_counts()['Match']
match_perc = round((Match_count/df.shape[0])*100, 2)


df_score = pd.DataFrame({'Match': [f"{Match_count}\{df.shape[0]}", f"{match_perc}%"]})


df_score.to_csv('Eval_comparison_score.csv')



