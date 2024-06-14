# %%
#https://www.youtube.com/watch?v=zRdzLfoTwvQ
import os
import openai
import json

with open('SETTINGS.json') as json_file:
    SETTINGS = json.load(json_file)['SETTINGS']

openai.api_key=SETTINGS["LLM"]["OPENAI"]["API_KEY"]

# %%
# completion=openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": "You are a assistant which informs about temperature."},
#         {"role": "user", "content": "Hey there"}
#     ]
# )


# %%

#print(completion.choices[0].message)


# %%

# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
import requests
def get_current_weather(location):
    """Get the current weather in a given location"""

    url = "https://ai-weather-by-meteosource.p.rapidapi.com/find_places"

    querystring = {"text":location}

    headers = {
      "X-RapidAPI-Key": "XXX",
      "X-RapidAPI-Host": "XXX"
    }

    response =f"current weather temp  of {location} is 27 deg f"
    print(response)
  
    return response
#response=get_current_weather('Bangalore')



# %%


# %%

functions = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    
                },
                "required": ["location"],
            },
        }
    ]



# %%

user_message="Hi There"
messages=[]
messages.append({"role": "user", "content":user_message})
# completion=openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=
#        messages
    
# )
# print(completion.choices[0].message)


# %%

messages


# %%

user_message="What is the temperature of Bangalore"

messages.append({"role": "user", "content": user_message})
# completion=openai.ChatCompletion.create(
#     model="gpt-3.5-turbo-0613",
#     messages=messages,
#     functions=functions
    
# )


# # %%

# messages

# completion


# %%

# print(completion.choices[0].message)

# response=completion.choices[0].message
# response

# function_name=response['function_call']['name']
# print(function_name)
# get_current_weather
import json
# location=eval(response['function_call']['arguments'])['location']
# print(location)

# Step 4: send the info on the function call and function response to GPT
# messages.append(response)  # extend conversation with assistant's reply
# messages.append(
#     {
#         "role": "function",
#         "name": function_name,
#         "content": location,
#     }
# )
# messages

# extend conversation with function response
# second_response = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo-0613",
#     messages=messages,
#     functions=functions
# )  # get a new response from GPT where it can see the function response
# print(second_response.choices[0].message)



# # %%

# second_response


# # %%

# messages


