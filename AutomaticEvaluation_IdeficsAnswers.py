import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
#from openai import OpenAI
import openai
import re


openai.api_key = "ca72e43668824cd090e6ea210fe3cd3b"
openai.api_base = "https://hlt-nlp.openai.azure.com/" 
openai.api_type = 'azure'
openai.api_version = '2023-05-15' # this may change in the future
deployment_name='gpt-35-turbo'



def GPT_answer_eval(question, caption, System_answer):
    response = openai.ChatCompletion.create(
        engine= deployment_name,
        messages=[
            {"role": "system", "content": f"You are an assistant designed to judge the correctness of sentences related to videos." 
            f"Users will paste in a string of text a question Q about a video, an answer A, representing the caption of a video, and a sentence S."
            f"You have to judge whether S is correct with reference to Q and A."
            f"If so, you must answer \"Correct\", otherwise type \"Not_correct\". "},
            {"role": "user", "content": f"Q: {question}\nA: {caption}\nS: {System_answer}"}
        ],
        max_tokens = 2
    )
    return response['choices'][0]['message']['content']
#Ho messo max_token a 2 perchè giustamnete quando fa 'Not_matching", tende a darmi anche la spiegazione del perchè




def evaluation_accuracy(df):

    #Correctness eval with caption
    Correct_count = df['System_evaluation'].value_counts()['Correct']
    NotMatching_count = df['System_evaluation'].value_counts()['Not_correct']
    correct_perc = round((Correct_count/df.shape[0])*100, 2)
    notMatching_perc = round((NotMatching_count/df.shape[0])*100, 2)



    
    df_acc = pd.DataFrame({'Correct': [f"{Correct_count}\{df.shape[0]}", f"{correct_perc}%"], 'Not_Correct' : [f"{NotMatching_count}\{df.shape[0]}", f"{notMatching_perc}%"] })
    return df_acc




df = pd.read_csv('DF_pilot_withCaption_Foil_Idefics.csv', index_col = 0)

print("Dataframe correctly loaded!")
print()


print("GPT is checking Idefics answers...")
print()


#GPT checking sentence similarity between Idefics answers and our manually created answers wrt each video
for index, row in df.iterrows():
    question = row['Question']
    System_answer = row['Idefics_Answer']
    caption = row['Caption']
    response = GPT_answer_eval(question, caption, System_answer)
    df.loc[index, 'System_evaluation'] = response.strip('.')
    print(index, caption, response)
    
print("All answers correctly checked and saved!")
print()




df.to_csv('/home/dtesta/MyProjects/Benchmark_PilotTest/DF_pilot_withAutomaticEvaluation.csv')
print("DF correctly updated!")
print()


df_acc = evaluation_accuracy(df)
print("DF with evaluation scores correctly created...")
print()

df_acc.to_csv('/home/dtesta/MyProjects/Benchmark_PilotTest/Idefics_Evaluation_GPT.csv')
print("... and saved!")


