"""
Use to get completions from an OpenAI completions endpoint. This version
of the script only works with Azure OpenAI service, since OpenAI no longer
hosts their code completion models.

source: https://github.com/nuprl/MultiPL-E/blob/main/openai_model.py
"""
import pdb
from pathlib import Path
from typing import List, Union, Dict, Optional
from completions import partial_arg_parser, make_main
import json
import openai

from rlhf_exp.sampling.util import convert_prompt_to_messages

try:
    from openai.error import RateLimitError
except:
    from openai import RateLimitError
import os
import time
from typing import List, Dict

global engine, model

openai.api_key_path=Path(__file__).parent.parent.joinpath("openai_bu_key.txt")
# openai.api_key_path=Path(__file__).parent.parent.joinpath("openai_my_gmail_key.txt")



def completions(
    prompts: List[str], max_tokens: int, temperature: float, top_p, stop
) -> List[str]:
    results = []
    for prompt in prompts:
        kwargs = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stop": stop
        }

        if engine is not None:
            kwargs["engine"] = engine
        elif model is not None:
            kwargs["model"] = model

        while True:
            try:
                result = openai.Completion.create(**kwargs)
                result = result["choices"][0]["text"]
                break
            except RateLimitError:
                print("Rate limited...")
                time.sleep(5)
            except Exception as e:
                print(f"Exception found: {e}")
                if "maximum context length" in str(e):
                    result = ""
                    break
                time.sleep(5)
        results.append(result)
        time.sleep(0.5)
    return results


def chat_completions(
    prompts: Union[List[str], List[List[Dict[str, str]]]], max_tokens: int, temperature: float, top_p, stop, completion_model=None
) -> List[str]:
    results = []
    for prompt in prompts:
        if isinstance(prompt, str):
            messages = convert_prompt_to_messages(prompt)
        else:
            messages = prompt
        kwargs = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        if isinstance(stop, list) and len(stop) > 0:
            kwargs["stop"] = stop

        if completion_model is not None:
            kwargs["model"] = completion_model
        elif engine is not None:
            kwargs["engine"] = engine
        elif model is not None:
            kwargs["model"] = model

        while True:
            try:
                result = openai.ChatCompletion.create(**kwargs)
                result = result["choices"][0]["message"]["content"]
                break
            except RateLimitError as e:
                print(f"Rate limited... {e}")
                time.sleep(5)
            except Exception as e:
                print(f"Exception found: {e}")
                if "maximum context length" in str(e):
                    result = ""
                    break
                time.sleep(5)
        results.append(result)
        time.sleep(0.5)
    return results


def main():
    global engine, model
    args = partial_arg_parser()
    args.add_argument("--model", type=str)
    args.add_argument("--engine", type=str)
    args.add_argument("--name-override", type=str)
    args.add_argument("--azure", action="store_true")
    args.add_argument("--chat-completion", action="store_true", help="Use this flag when running chat completions.")
    args = args.parse_args()

    if args.engine is None and args.model is None:
        raise ValueError("Must specify either engine or model.")
    elif args.engine is not None and args.model is not None:
        raise ValueError("Must specify either engine or model, not both.")

    engine = args.engine
    model = args.model
    if args.azure:
      openai.api_type = "azure"
      openai.api_base = os.getenv("OPENAI_API_BASE")
      openai.api_version = "2022-12-01"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if args.name_override:
        name = args.name_override
    else:
        if args.engine is not None:
            name = args.engine
        else:
            name = args.model

    make_main(args, name, chat_completions if args.chat_completion else completions)


if __name__ == "__main__":
    main()