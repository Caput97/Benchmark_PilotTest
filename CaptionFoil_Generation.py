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



def create_captionGPT(question, answer):
    response = openai.ChatCompletion.create(
        engine= deployment_name, # The deployment name you chose when you deployed the GPT-3.5-Turbo or GPT-4 model.
        messages=[
            {"role": "system", "content": f"You are an assistant designed to create declarative sentences." 
            f"Users will paste in a string of text a question Q and an answer A which are related to a video scene. Based on Q and A, you have to create "
            f"a sentence that is the caption for the video. Remember: the caption is a declarative sentence but it must be short and can contain only what is said in Q and A!"},
            {"role": "user", "content": f"Q: {question}\nA: {answer}"}
        ]
    )
    return response.choices[0].message.content




def create_foilGPT(caption):
    response = openai.ChatCompletion.create(
        engine= deployment_name, # The deployment name you chose when you deployed the GPT-3.5-Turbo or GPT-4 model.
        messages=[
            {"role": "system", "content": f"You are an assistant designed to create foils for video captions." 
            f"Users will paste in a string of text a sentence C that is the caption of a video. Based on C, you must create only one foil F for that caption. Don't type any other information." 
            f"Remember: A foil differs from the caption for only one element!"},
            {"role": "user", "content": f"C: {caption}"}
        ]
        
    )
    output= response

    while output['choices'][0]['finish_reason'] == 'content_filter':
        response = openai.ChatCompletion.create(
            engine= deployment_name, # The deployment name you chose when you deployed the GPT-3.5-Turbo or GPT-4 model.
            messages=[
                {"role": "system", "content": f"You are an assistant designed to create foils for video captions." 
                f"Users will paste in a string of text a sentence C that is the caption of a video. Based on C, you must create only one foil F for that caption. Don't type any other information." 
                f"Remember: A foil differs from the caption for only one element!"
                #f"Ensure that the generated content is appropriate and follows OpenAI's usage policies."
                },
                {"role": "user", "content": f"C: {caption}"}
            ]
        
        )
        output= response

    

    while not re.findall('^F: .*|^Foil: .*', output['choices'][0]['message']['content']):
        response = openai.ChatCompletion.create(
            engine= deployment_name, # The deployment name you chose when you deployed the GPT-3.5-Turbo or GPT-4 model.
            messages=[
                {"role": "system", "content": f"You are an assistant designed to create foils for video captions." 
                f"Users will paste in a string of text a sentence C that is the caption of a video. Based on C, you must create only one foil F for that caption. Don't type any other information." 
                f"Remember: A foil differs from the caption for only one element!"},
                {"role": "user", "content": f"C: {caption}"}
            ]
        
        )
        output= response



    return output['choices'][0]['message']['content']



print("Loading dataframe...")
print()

#load the df
df = pd.read_csv('/home/dtesta/MyProjects/Benchmark_PilotTest/DF_pilot.csv', index_col = 0)


print("Dataframe correctly loaded...")
print()




print("Generating captions...")
print()


#GPT generating captions based on Q+A
for index, row in df.iterrows():
    question = row['Question']
    answer = row['Answer']
    caption = create_captionGPT(question, answer)
    df.loc[index, 'Caption'] = caption
    
print("Captions correctly generated!")
print()




print("Generating foils...")
print()


#GPT generating foils based on captions
for index, row in df.iterrows():

    caption = row['Caption']
    foil = create_foilGPT(caption)
    df.loc[index, 'Foil'] = foil


print("Foils correctly generated!")
print()




print("Cleaning up outputs...")
print()

#clean up noise from caption_outputs
for index, row in df.iterrows():

    cleaned_caption = re.sub(r'Caption:|"|Sentence:|Caption suggestion:', '', row['Caption'])  
    df.loc[index, 'Caption'] = cleaned_caption

    cleaned_foil = re.sub(r'\w* foil:|Foil\d*:|"|F:', '', row['Foil'])  
    df.loc[index, 'Foil'] = cleaned_foil

print("Outputs correctly cleaned!")
print()



df.to_csv('/home/dtesta/MyProjects/Benchmark_PilotTest/DF_pilot_withCaptionFoils.csv')

print("Dataframe correctly updated!")