import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
#from openai import OpenAI
import openai
import re


openai.api_key = "ca72e43668824cd090e6ea210fe3cd3b"
openai.api_base = "https://hlt-nlp.openai.azure.com/" # your endpoint should look like the following
openai.api_type = 'azure'
openai.api_version = '2023-05-15' # this may change in the future
deployment_name='gpt-35-turbo'



def GPT_answer_eval(System_answer, answer):
    response = openai.ChatCompletion.create(
        engine= deployment_name, # The deployment name you chose when you deployed the GPT-3.5-Turbo or GPT-4 model.
        messages=[
            {"role": "system", "content": f"You are an assistant designed to assess the similarity of sentences." 
            f"Users will paste in a string of text both the caption of a video, defined as C, and a declarative sentence S. "
            f"Given C and S, you must check whether they are semantically similar and whether they are expressing the same concept."
            f"If so, you must answer \"Correct\", otherwise type \"Not_correct\". "},
            {"role": "user", "content": f"C: {System_answer}\nS: {answer}"}
        ],
        max_tokens = 2
    )
    return response['choices'][0]['message']['content']

def evaluation_accuracy(df):
    Correct_count = df['System_evaluation'].value_counts()['Correct']
    NotMatching_count = df['System_evaluation'].value_counts()['Not_correct']
    correct_perc = round((Correct_count/df.shape[0])*100, 2)
    notMatching_perc = round((NotMatching_count/df.shape[0])*100, 2)
    
    df_acc = pd.DataFrame({'': ['Count', 'Percentage'], 'Correct': [f"{Correct_count}\{df.shape[0]}", f"{correct_perc}%"], 'Not_Matching' : [f"{NotMatching_count}\{df.shape[0]}", f"{notMatching_perc}%" ]})
    return df_acc




#Ho messo max_token a 2 perchè giustamnete quando fa 'Not_matching", tende a darmi anche la spiegazione del perchè



df = pd.read_csv('DF_pilot_withCaption_Foil_Idefics.csv', index_col = 0)

print("Dataframe correctly loaded!")
print()




print("GPT is checking Idefics answers...")
print()


#GPT checking sentence similarity between Idefics answers and our manually created answers wrt each video
for index, row in df.iterrows():
    System_answer = row['Idefics_Answer']
    answer = row['Answer']
    response = GPT_answer_eval(System_answer, answer)
    df.loc[index, 'System_evaluation'] = response.strip('.')
    
print("All answers correctly checked and saved!")
print()


df.to_csv('/home/dtesta/MyProjects/Benchmark_PilotTest/DF_pilot_withAutomaticEvaluation.csv')
print("DF correctly saved!")
print()


df_acc = evaluation_accuracy(df)
print("DF with evaluation accuracy correctly created...")
print()

df_acc.to_csv('/home/dtesta/MyProjects/Benchmark_PilotTest/Idefics_Evaluation_GPT.csv')
print("... and saved!")




