import openai
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import re
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import json

openai.api_key = "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"

keywords = [
    "Parachuting",
    "Origami",
    "Skateboarding",
    "Fireworks",
    "Beekeeping",
    "Street Art",
    "Sandcastles",
    "Violin Covers",
    "Space Exploration",
    "Dog Training",
    "Robotics",
    "Surfing",
    "Bubble Tea",
    "Mountain Biking",
    "Chess Tournaments",
    "Urban Gardening",
    "Magic Tricks",
    "Calligraphy",
    "Aerial Yoga",
    "Underwater Photography",
    "Humanoid",
    "Sushi",
    "Parkour",
    "Dancing",
    "Playing basketball"
]

# Shortened keywords for testing
keywords = [    
    "Dog Training"
]

# META PROMPT GENERATION

def generate_meta_prompt(keyword, model='gpt-4o', temp=0.7):
    
    system = """
        You are an assistant for generating meta prompts that help a large language model (LLM) produce short, vivid, and grounded prompts for retrieving or generating video content.


        The goal is **not to write full storylines or scripts**, but to guide the LLM to create **visually-descriptive prompts** that describe what might be seen in short video clips.


        Meta prompts should:
        - Focus on describing visual scenes.
        - Mention actions, settings, objects, and participants.
        - Be open-ended enough to allow variety, but grounded enough to encourage realistic or plausible content.


        Here is an example:


        Keyword: 'Humanoid'
        Meta prompt: 'Generate visually grounded prompts describing humanoid robots performing real-world tasks alongside humans in everyday environments.'


        Avoid narrative storytelling. Stay concise and descriptive."""


    prompt = f"Given the keyword {keyword}, generate 5 meta prompts that help describe short, visually-grounded video scenes involving {keyword}. Each meta prompt should be 1-2 sentences. Separate each meta prompt with a newline."

    model="gpt-4o"
    
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temp,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    response_text = response.choices[0].message.content.strip()
    lines = re.split(r'\n?\d+\.\s+', response_text.strip())
    meta_prompts = [line.strip() for line in lines if line.strip()]
    return meta_prompts

def generate_prompts(meta_prompt, model='gpt-4o', temp=0.7):
    system = """
        You are an assistant that generates short, specific, and visually-grounded prompts for video retrieval or generation.


        You will be given a meta prompt that describes a visual theme. Your task is to generate 5–10 actual prompts that:
        - Are short (1 sentence each)
        - Describe **realistic, concrete visual scenes**
        - Include visible elements like people, actions, settings, and objects
        - Are suitable for generating or retrieving short video clips


        Avoid storytelling, character names, abstract concepts, or hypotheticals. Focus on scenes that can be clearly visualized.


        ---


        Here are two examples:


        Meta prompt:
        "Generate prompts describing humanoid robots performing real-world tasks alongside humans in everyday environments."


        Actual prompts:
        1. A humanoid robot helping a person carry grocery bags in a parking lot.
        2. A humanoid robot preparing lunch in a family kitchen.


        Meta prompt:
        "Generate prompts describing people dancing in various cultural or natural settings."


        Actual prompts:
        1. A group of friends dancing barefoot at the beach during sunset.
        2. Traditional dancers performing in colorful costumes at a street festival.


        ---


        Now, use this format to generate actual prompts from new meta prompts.
        """



    prompt =  f"""
        Here is a meta prompt:
        {meta_prompt}

        Generate 5 actual prompts, each 1 sentence long, describing visually-grounded scenes.
        """


    model="gpt-4o"
    
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temp,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    response_text = response.choices[0].message.content.strip()
    lines = re.split(r'\n?\d+\.\s+', response_text.strip())
    prompts = [line.strip() for line in lines if line.strip()]
    return prompts

# BLEU SCORE CALCULATION (Scaled to [0, 0.5]. The higher the more diverse)
def scaled_self_bleu(prompts):
    smoothing = SmoothingFunction().method1
    scores = []
    for i in range(len(prompts)):
        hypo = prompts[i].split()
    refs = [gen.split() for j, gen in enumerate(prompts) if j != i]
    score = sentence_bleu(refs, hypo, weights=(1.0, ), smoothing_function=smoothing)
    scores.append(score)
    return 0.5 - sum(scores) / (2 * len(scores))

# Agentic Diversity Evaluator (scaled to [0, 0.5]. The higher the more diverse)
def diversity_criticizer(prompts, meta_prompt):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful prompt critic that generates a score of how semantically diverse the generated set of prompts are to the meta prompt"},
            {"role": "user", "content": f"Give a score from 0 to 0.5 of how semantically diverse the following prompt is to the meta prompt with two decimals: {meta_prompt}, the prompts are: {prompts}. Only output the score."}
        ],
        temperature=0.7,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return float(response.choices[0].message.content)

# Final Diversity Score (self_bleu + diversity_criticizer)
def diversity_score(prompts, meta_prompt):
    self_bleu = scaled_self_bleu(prompts)
    diversity = diversity_criticizer(prompts, meta_prompt)
    return self_bleu + diversity

# Faithfulness Score (scaled to [0, 1]. The higher the more faithful)
def faithfulness_criticizer(prompts, meta_prompt):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful prompt critic that generates a score of how faithful the generated set of prompts are to the meta prompt"},
            {"role": "user", "content": f"Give a score from 0 to 1 of how faithful the following prompt is to the meta prompt with two decimals: {meta_prompt}, the prompts are: {prompts}. Only output the score."}
        ],
        temperature=0.7,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return float(response.choices[0].message.content)

# Final Prompt Score (faithfulness * diversity)
def prompt_score(prompts, meta_prompt):
    faithfulness = faithfulness_criticizer(prompts, meta_prompt)
    diversity = diversity_score(prompts, meta_prompt)
    return faithfulness * diversity

if __name__ == "__main__":

    prompts = {keyword: {meta_prompt: generate_prompts(meta_prompt) for meta_prompt in tqdm.tqdm(generate_meta_prompt(keyword))} for keyword in keywords}
    
    for keyword, meta_prompts in prompts.items():
        for meta_prompt, generated_prompts in tqdm.tqdm(meta_prompts.items()):
            score = prompt_score(generated_prompts, meta_prompt)
            while score < 0.45:
                generated_prompts = generate_prompts(meta_prompt)
                score = prompt_score(generated_prompts, meta_prompt)

    with open("generated_prompts.json", "w") as f:
        json.dump(prompts, f, indent=2)