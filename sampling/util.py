import string
from pathlib import Path
import json
import gzip
from typing import Optional, List, Dict, Tuple, Union, Any
import sys
import regex as re

import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords

import pandas as pd

from prompts import *
from openai_model import completions, chat_completions

SYSTEM_INIT = "<<<SYSTEM>>>"
USER_INIT = "<<<USER>>>"
ASSISTANT_INIT = "<<<ASSISTANT>>>"
RESPONSE_SAMPLING_KWARGS = {
    "max_tokens": 512,
    "temperature": 1.0,
    "top_p": 1.0,
    "stop": None,
    "completion_model": "gpt-4-0613"
}
AUTOMODEL_RESPONSE_SAMPLING_KWARGS = {
    "max_tokens": 512,
    "temperature": 1.0,
    "top_p": 1.0,
    "stop": [],
}

def query_questions(prompt: str) -> List[str]:
    """
    Given a prompt query (sampling X) OpenAI model to obtain a list of questions for this persona

    Args:
        prompt (str): prompt query

    Returns:
        List[str]: list of questions
    """
    qs = []
    completions = chat_completions(
        prompts=[prompt] if isinstance(prompt, str) else prompt,
        **RESPONSE_SAMPLING_KWARGS
    )
    for completion in completions:
        questions = completion.strip().split("\n")

        for q in questions:
            q = q.strip()
            if q.endswith('"'):
                parsed_q = q[q.find('"') + 1:-1]
            elif q.endswith('”'):
                parsed_q = q[q.find('“') + 1:-1]
            elif "." in q[:3]:
                parsed_q = q[q.find(".") + 1:]
            else:
                parsed_q = q
            if not parsed_q.strip():
                continue
            qs.append(parsed_q.strip())
    return qs


def get_category(name, axis, persona_df, include_names: bool = True):
    axis_list = persona_df[persona_df.name == name].axis.tolist()[0]
    specific_axis = [a for a in axis_list if axis in a][0]
    if include_names:
        name_category = specific_axis.replace(f"{axis}: ", f"{name}: ").replace(": ", " (") + ")"
        return name_category
    else:
        category = specific_axis[specific_axis.find(": ") + 2:]
        return category


def join_names(names: List[str]) -> str:
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)


def is_whole_word_in(w, s):
    if re.search(r"\b" + re.escape(w) + r"\b", s):
        return True
    return False


def merge_persona_axis(df: pd.DataFrame) -> pd.DataFrame:
    personas = {}
    df["axis"] = df.apply(lambda r: [f"{r.axis}: {r.category}"], 1)
    df["description"] = df.description.apply(lambda x: [x])
    del df["category"]
    for i, row in df.iterrows():
        if row['name'] not in personas:
            personas[row['name']] = row.to_dict()
        else:
            personas[row['name']]["axis"].extend(row.axis)
            personas[row['name']]["description"].extend(row.description)

    persona_rows = [{"name": k, **v} for k, v in personas.items()]
    out_df = pd.DataFrame(persona_rows)
    return out_df


def extract_persona_name(persona: Optional[str]) -> str:
    return persona[:persona.find(",")] if persona is not None else None



def stop_at_stop_token(decoded_string, stop_tokens):
    """
    Produces the prefix of decoded_string that ends at the first occurrence of
    a stop_token.

    WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
    itself.
    """
    min_stop_index = len(decoded_string)
    for stop_token in stop_tokens:
        stop_index = decoded_string.find(stop_token)
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    return decoded_string[:min_stop_index]


def remove_empty_responses(df, n_responses):
    for i, row in df.iterrows():
        non_empty_completions_indices = [1 if len(y) > 1 else 0 for y in row.y_cot]
        if sum(non_empty_completions_indices) < n_responses:
            #  remove any empty completions if exist, move both cot and y the same time to ensure consistency
            df.at[i, "y"] = [y for i, y in enumerate(row.y) if non_empty_completions_indices[i]]
            df.at[i, "y_cot"] = [y for i, y in enumerate(row.y_cot) if non_empty_completions_indices[i]]



def generate_completions_with_gold_categories(prompts: List[str], df: pd.DataFrame, df_persona: pd.DataFrame) -> List[str]:
    ys = []
    for prompt in prompts:
        x = prompt.split("<|user|>")[-1].replace("<|assistant|>", "").replace("</s>", "").strip()
        if "axis" in df.columns: # common questions
            axis = df[df.prompt.str.strip() == x].axis.tolist()[0]
        else:  # personal question, randomly pick
            name = df[df.prompt.str.strip() == x].name.tolist()[0]
            axis = np.random.choice(df_persona[df_persona.name == name].axis.tolist()[0]).split(": ")[0]
        sub_persona_df = df_persona[df_persona.axis.apply(lambda s: f"{axis}: " in str(s))]
        categories = [get_category(name, axis, sub_persona_df, include_names=False) for name in sub_persona_df.name]
        category_str = ", ".join(categories)
        y = f"Axis: {axis}\nCategories: {category_str}\nChosen category: {categories[0]}\nResponse: dummy response"
        ys.append(y)
    return ys




def get_y_from_cot_y(y):
    max_end = max(y.lower().find("chosen category:"), y.lower().find("chosen subcategory:"))
    if y.lower().find("response:") > max_end:
        max_end = max_end if max_end > -1 else 0
        response = y[y.lower().rfind("response:", max_end) + 10:].strip()
    else:
        response = y[y.find("\n", max_end):].strip()
    return response


def trim_y(y):
    if y.startswith(":"):
        y = y[1:].strip()
    for phrase in ["axis:", "categories:", "category:"]:
        if phrase in y[:100].lower():
            nextline = y.find("\n", y.lower().find(phrase))
            if nextline < 200:
                y = y[nextline:].strip()
    return y.strip()


def get_cot(y):
    for phrase in ["chosen category:", "chosen subcategory:"]:
        idx = y.lower().find(phrase)
        if idx != -1:
            return y[:idx].strip()
    return ""


def extract_cot_and_response(ys):
    responses = []
    cots = []
    for y in ys:
        # if starts to generate new set of categories after response, trim off
        for key_phrase in ["\naxis:", "\ncategories:", "\nchosen category:", "\nresponse:"]:
            key_idx = y.lower().find(key_phrase)
            if key_idx > 650:
                y = y[:key_idx]
        # here we are assuming chose category is always generated, but responses header might not be
        y_cot = get_cot(y)
        cots.append(y_cot)
        response = get_y_from_cot_y(y) if y_cot or "response:" in y.lower() else y.strip()
        trimed_response = heuristics_modifier(response)
        response = trimed_response if len(trimed_response) > 100 else response
        responses.append(response)
    return cots, responses


def last_occurance(s: str, phrase_list: List[str], end: Optional[int] = None) -> int:
    end = len(s) if end is None else end
    return max([s.rfind(p, 0, end) if s.rfind(p, 0, end)!=-1 else -100000 for p in phrase_list])


def heuristics_modifier(y: str):
    for _ in range(2):
        y = y.strip()
        if len(y) == 0:
            y = "empty response :("
        if y[0] == ":":
            y = y[1:].strip()
        y = trim_y(y)

        if any([y.lower().startswith(phrase) for phrase in ["as", "for", "given", "based", "in response", "thus", "in line"]]):
            # example: "As someone committed to practicing mindful eating... , you've..."
            # example: "For individuals prioritizing mental health ..., I'd suggest"
            # example: ": Given your interest in attending film festivals, as someone from ... of Australia, you m
            # here we want to remove the first subjunctive clause, and start with "You've"
            y = capitalize(y[y.find(",")+2:].strip())
        elif any([y.lower().startswith(phrase) for phrase in ["respon", "chosen"]]):
            # example: "Response for someone struggling with negative thought patterns:"
            # example: "Chosen Geographical Location (South America) and Festival category (Narrative feature films):"
            # here, we want to find the next :.\n the comes first and truncate till then
            min_idx = first_occurance(y, [':','.','\n'])
            min_idx = -2 if min_idx == 100000 else min_idx
            y = y[min_idx+2:].strip()
        elif any([y.lower().startswith(phrase) for phrase in ["dear", "hi", "hello", "however", "greeting", "fellow"]]):
            # example: ": Dear user, who I assume may have recently ..., let me elaborate on some techniques"
            # example: "Hi [User Name]!\n\nAs someone interested in the..."
            # here, we remove first comma within first 50 characters (usually about some assumptions of
            # user profile), then remove up to first comma, or end of sentence
            if "," in y[:50]:
                y.replace(",", "", 1)
            min_idx = first_occurance(y, [',', '.','!',':','\n'])
            min_idx = -2 if min_idx == 100000 else min_idx
            y = capitalize(y[min_idx + 2:].strip())
        elif y.startswith("("):
            # example: "(for a potential attendee based in Vietnam)"
            # remove the whole parenthesis
            y = y[y.find(")")+1:].strip()

        for phrase in ["chosen category", "letting me know", "like you" "as someone", "as a ", " fellow ", "dear ",
                       "preference","preferred", "orientation", "identif", " cater", " previous", " specified ",
                       "selected", "axis" ," geared ", " friend", "sorry"]:
            if in_beginning(y, phrase, 400):
                # example: Laughter has the amazing capability.., providing your chosen category for benefits - the ...
                # example: the native population. As someone who belongs to the ..., ... high-demand professions. Accor
                # example:
                # try deleting the whole sentence.
                y = remove_containing_sentence(y, phrase)
    return y




def capitalize(s: str):
    if len(s) == 0:
        return s
    if len(s) == 1:
        return s.upper()
    return s[0].upper() + s[1:]


def in_beginning(s: str, phrase: str, by: int=100):
    return phrase.lower() in s.lower() and s.lower().find(phrase.lower()) < by


def first_occurance(s: str, phrase_list: List[str], start: int=0) -> int:
    return min([s.find(p, start) if s.find(p, start)!=-1 else 100000 for p in phrase_list])


def remove_containing_sentence(y, phrase):
    phrase_idx = y.lower().find(phrase)
    sentence_end = first_occurance(y, ['.', '!', ':', '\n', '?'], start=phrase_idx)
    sentence_end = -2 if sentence_end == 100000 else sentence_end
    sentence_start = last_occurance(y, ['.', '!', ':', '\n', '?'], end=phrase_idx)
    sentence_start = -1 if (sentence_start == -100000) else sentence_start
    y = f"{y[:sentence_start+1].strip()} {y[sentence_end + 2:].strip()}".strip()
    return y


def format_persona_inference_prompt(
    train_df: pd.DataFrame,
    persona_df: pd.DataFrame,
    format_to_chat:bool,
    name: str,
    num_shots: int,
    opinion_qa_prompt: bool,
    reveal_name: bool,
    x_only: bool,
    seed: Optional[int]=None,
    data_indices: Optional[List[int]]=None
):
    if "yw" not in train_df.columns:
        train_df["yw"] = train_df.chosen.apply(lambda x: x[-1]["content"])
        train_df["yl"] = train_df.rejected.apply(lambda x: x[-1]["content"])

    name_df = train_df[train_df.name == name]
    if data_indices is not None:
        join_qs = name_df.reindex().iloc[data_indices].reset_index()
    elif seed is not None:
        join_qs = name_df.sample(n=min(num_shots, len(name_df)), random_state=seed).reset_index()
    else:
        join_qs = name_df.sample(n=min(num_shots, len(name_df))).reset_index()

    if x_only:
        qs = [q.strip() for q in join_qs.prompt.tolist()]
        prompt_template = SAMPLE_PERSONA_GIVEN_X
        prompt = prompt_template.format(X_LIST="- " + "\n- ".join(qs))
    elif not reveal_name:
        prompt_template = SAMPLE_PERSONA_GIVEN_X_Y_OQA if opinion_qa_prompt else SAMPLE_PERSONA_GIVEN_X_Y
        xy_list = ""
        for i, row in join_qs.iterrows():
            if opinion_qa_prompt:
                xy_list += f"## Survey Question {i + 1}:\n{row.prompt}\n### Preferred Choice:\n{row.yw}\n### Dispreferred Choice:\n{row.yl}\n\n"
            else:
                xy_list += f"## User Question {i + 1}:\n{row.prompt}\n### Preferred Response:\n{row.yw}\n### Dispreferred Response:\n{row.yl}\n\n"
        prompt = prompt_template.format(XY_LIST=xy_list.strip() + "\n")
    else:  # gold persona, reveal person's name
        prompt_template = SAMPLE_PERSONA_GIVEN_NAME
        prompt = prompt_template.format(NAME=name)

    if format_to_chat:
        system_msg = prompt.split('<<<USER>>>')[0].replace("<<<SYSTEM>>>", "")
        user_msg = prompt.split('<<<USER>>>')[1].replace("<<<USER>>>", "")
        prompt = {"messages": [{"content": system_msg, "role": "system"},
                               {"content": user_msg, "role": "user"}]}
    return prompt



def convert_prompt_to_messages(prompt: str) -> List[Dict[str, str]]:
    assert prompt.startswith(SYSTEM_INIT), "No system message provided for chat completion prompt"
    system_message = prompt[:prompt.find(USER_INIT)]
    messages = [{"role": "system", "content": system_message.replace(SYSTEM_INIT, "").strip()}]
    rest_prompt = prompt.replace(system_message, "")
    while rest_prompt.startswith(USER_INIT) or rest_prompt.startswith(ASSISTANT_INIT):
        user_turn = rest_prompt.startswith(USER_INIT)
        if user_turn:
            next_turn_idx = rest_prompt.find(ASSISTANT_INIT)
            next_turn_content = rest_prompt if next_turn_idx == -1 else rest_prompt[:next_turn_idx]
            next_turn_content = next_turn_content.replace(USER_INIT, "")
            role = "user"
        else:
            next_turn_idx = rest_prompt.find(USER_INIT)
            next_turn_content = rest_prompt if next_turn_idx == -1 else rest_prompt[:next_turn_idx]
            next_turn_content = next_turn_content.replace(ASSISTANT_INIT, "")
            role = "assistant"
        rest_prompt = rest_prompt[next_turn_idx:]
        messages.append({"role": role, "content": next_turn_content.strip()})
    return messages


def deepseek_chat_completions(prompts: List[str], **completion_kwargs):
    from openai import OpenAI
    client = OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com")
    results = []
    for prompt in prompts:
        if isinstance(prompt, str):
            messages = convert_prompt_to_messages(prompt)
        else:
            messages = prompt
        kwargs = {
            "messages": messages,
            "model":"deepseek-chat",
            "stream":False,
            **completion_kwargs,
        }
        response = client.chat.completions.create(**kwargs)
        results.append(response.choices[0].message.content)
    return results
