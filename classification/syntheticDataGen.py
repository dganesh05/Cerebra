import openai
import pandas as pd

client = openai.OpenAI(api_key = 'key')

def generate_message(prompt, n=3, temperature=1.2):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant in creating diverse and casual messages to train a support chatbot."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": "Ensure that the messages are natural, friendly, and resemble everyday online conversations. Remember"
            "that greetings are not required."}
        ],
        max_tokens=50,
        temperature=temperature,
        n=n,
    )
    return [choice.message.content.strip() for choice in completion.choices]

data = []

for _ in range(2):
    messages_0 = generate_message("Generate an informal Discord message that could not be answered by a bot using information "
    "documents provided by the Hackathon organizers.")
    for msg in messages_0:
        data.append((msg, 0))
    
    messages_1 = generate_message("Generate an informal Discord message that could be answered by a bot using information "
    "documents provided by the Hackathon organizers.")
    for msg in messages_1:
        data.append((msg, 1))

df = pd.DataFrame(data, columns=['Message', 'Response'])

print(df.head())

df.to_csv('discord_synthetic_data_llm.csv', index=False)
