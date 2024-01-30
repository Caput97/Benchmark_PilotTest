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



def GPT_answer_eval(System_answer, caption):
    response = openai.ChatCompletion.create(
        engine= deployment_name,
        messages=[
            {"role": "system", "content": f"You are an assistant designed to assess the similarity of sentences." 
            f"Users will paste in a string of text both the caption of a video, defined as C, and a declarative sentence S."
            f"Given C and S, you must check whether they are semantically similar and whether they are expressing the same concept."
            f"If so, you must answer \"Correct\", otherwise type \"Not_correct\". "},
            {"role": "user", "content": f"C: {System_answer}\nS: {caption}"}
        ],
        max_tokens = 2
    )
    return response['choices'][0]['message']['content']
#Ho messo max_token a 2 perchè giustamnete quando fa 'Not_matching", tende a darmi anche la spiegazione del perchè




def GPT_NLI_Foil(System_answer, foil):
    response = openai.ChatCompletion.create(
        engine= deployment_name, 
        messages=[
            {"role": "system", "content": f"You are an assistant designed to act as a Natural Language Inference system." 
            f"Users will paste in a string of text two declarative sentences: C and S."
            f"Given C and S, you must test whether the two sentences express the same concept or not."
            f"If the sentences are different from a semantic point of view and contradictory, you must write \"Contradiction\", "
            f"otherwise if they express the same concept or a very similar one, you must write \"Not_contradiction\". "},
            {"role": "user", "content": f"C: {System_answer}\nS: {foil}"}
        ],
        max_tokens = 4
    )
    return response['choices'][0]['message']['content']
#4 token perché la parola contradiction la scorpora in 3 token diversi




def evaluation_accuracy(df):

    #Correctness eval with caption
    Correct_count = df['System_evaluation'].value_counts()['Correct']
    NotMatching_count = df['System_evaluation'].value_counts()['Not_correct']
    correct_perc = round((Correct_count/df.shape[0])*100, 2)
    notMatching_perc = round((NotMatching_count/df.shape[0])*100, 2)


    #Double check with Foil-condradiction
    ok_count = df['Double_check'].value_counts()['ok']
    notOk_count = df['Double_check'].value_counts()['not_ok']
    ok_perc = round((ok_count/df.shape[0])*100, 2)
    notOk_perc = round((notOk_count/df.shape[0])*100, 2)

    
    df_acc = pd.DataFrame({'Correct': [f"{Correct_count}\{df.shape[0]}", f"{correct_perc}%"], 'Not_Correct' : [f"{NotMatching_count}\{df.shape[0]}", f"{notMatching_perc}%"], 
                           'Double_check_ok': [f"{ok_count}\{df.shape[0]}", f"{ok_perc}%"], 'Double_check_no': [f"{notOk_count}\{df.shape[0]}", f"{notOk_perc}%"]  })
    return df_acc




df = pd.read_csv('DF_pilot_withCaption_Foil_Idefics.csv', index_col = 0)

print("Dataframe correctly loaded!")
print()


print("GPT is checking Idefics answers...")
print()


#GPT checking sentence similarity between Idefics answers and our manually created answers wrt each video
for index, row in df.iterrows():
    System_answer = row['Idefics_Answer']
    caption = row['Caption']
    response = GPT_answer_eval(System_answer, caption)
    df.loc[index, 'System_evaluation'] = response.strip('.')
    
print("All answers correctly checked and saved!")
print()


df.to_csv('/home/dtesta/MyProjects/Benchmark_PilotTest/DF_pilot_withAutomaticEvaluation.csv')
print("DF correctly saved!")
print()


print('Let\'s make a double check between Idefics\' answers and the foils...')
print()
#contradiction check with foils
for index, row in df.iterrows():
    System_answer = row['Idefics_Answer']
    foil = row['Foil']
    response = GPT_NLI_Foil(System_answer, foil)
    df.loc[index, 'System_evaluation_foil'] = response.strip('.')



print('... ')
print()

#double check evaluation
for index, row in df.iterrows():

    if row['System_evaluation'] == 'Correct' and row['System_evaluation_foil'] == 'Contradiction':
        df.loc[index, 'Double_check'] = 'ok'
    elif row['System_evaluation'] == 'Not_correct' and row['System_evaluation_foil'] == 'Not_contradiction':
        df.loc[index, 'Double_check'] = 'ok'
    else:
        df.loc[index, 'Double_check'] = 'not_ok'



print('Check done!')
print()



df_acc = evaluation_accuracy(df)
print("DF with evaluation accuracy correctly created...")
print()

df_acc.to_csv('/home/dtesta/MyProjects/Benchmark_PilotTest/Idefics_Evaluation_GPT.csv')
print("... and saved!")


