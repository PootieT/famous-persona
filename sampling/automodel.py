"""
This script produces completions for roughly any AutoModelForCausalLM.

source: https://github.com/nuprl/MultiPL-E/blob/main/automodel.py
"""
import pdb
from typing import List, Dict

from peft import PeftModel

from completions import make_main, partial_arg_parser
from util import stop_at_stop_token
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

class Model:
    def __init__(self, name, revision, tokenizer_name=None, tokenizer_revision=None):
        dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(name, revision=revision, torch_dtype=dtype,
                                                          trust_remote_code=True, device_map="auto" if torch.cuda.is_available() else None)
        # if torch.cuda.is_available():  # mostly for cpu debugging
        #     self.model = self.model.cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or name, revision=tokenizer_revision or revision,
                                                       padding_side="left", trust_remote_code=True)
        self.tokenizer.pad_token = "<|endoftext|>"

    def merge_adaptor(self, adaptor_path: str):
        print("loading adaptor and merging weights ...")
        model = PeftModel.from_pretrained(self.model, adaptor_path)
        model = model.merge_and_unload(progressbar=True)
        self.model = model

    def completion_tensors(
        self,
        prompts: List[str],
        max_length: int,
        temperature: float,
        top_p: float,
    ):
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt", return_token_type_ids=False).to(self.model.device)
        with torch.no_grad():
            # model_max_len = self.tokenizer.model_max_length # self.model.config.n_positions if "n_positions" in self.model.config else
            # actual_max_length = min(max_length, inputs["input_ids"].shape[1] - model_max_len)
            # if actual_max_length != max_length:
                # print(f"max_length reduced to {actual_max_length} from {max_length} due to model sequence limit.")
            output = self.model.generate(
                **inputs,
                do_sample=True,
                use_cache=True,
                top_p=top_p,
                temperature=temperature,
                # max_length=max_length,
                max_new_tokens=max_length, #actual_max_length,
                pad_token_id=self.tokenizer.pad_token_id
            )
        return output

    def decode_single_output(self, output_tensor, prompt):
        # NOTE(arjun): skip_special_tokens=True is the convenient way to strip out the left-side
        # padding tokens.
        detok_hypo_str = self.tokenizer.decode(
            output_tensor, clean_up_tokenization_spaces=False, skip_special_tokens=True,
        )
        # Skip the prompt (which may even have stop_tokens)
        # sometimes when prompt contain special characters the length changes before and after encode/decode
        decode_encode_prompt = self.tokenizer.decode(self.tokenizer(prompt)["input_ids"],
                                                     clean_up_tokenization_spaces=False, skip_special_tokens=True)
        # return detok_hypo_str[len(prompt):]
        return detok_hypo_str[len(decode_encode_prompt):]

    def completions(
        self, prompts: List[str], max_tokens: int, temperature: float, top_p, stop
    ):
        prompts = [prompt.strip() for prompt in prompts]
        output_tokens = []
        bs = len(prompts)
        i = 0
        while len(output_tokens) < len(prompts) or i < len(prompts):
            batch_prompts = prompts[i:i + bs]
            try:
                output_tensors = self.completion_tensors(
                    batch_prompts,
                    max_tokens,
                    temperature,
                    top_p,
                )
                tokens = [
                    stop_at_stop_token(self.decode_single_output(output_tensor, prompt), stop + [self.tokenizer.pad_token])
                    for (prompt, output_tensor) in zip(batch_prompts, output_tensors)
                ]
                output_tokens.extend(tokens)
                i += bs
            except Exception as e:
                print(f"Generation Error: {e}\n"
                      f"# characters of prompts: {[len(p) for p in batch_prompts]}\n"
                      f"# words of prompts: {[len(p.split()) for p in batch_prompts]}\n")
                if bs > 1:
                    print(f"try reducing batchsize from {bs} to {bs//2}")
                    bs = bs // 2
                else:
                    print(f"batch size is 1 and still OOD, skipping this datapoint")
                    output_tokens.append("")
                    i += 1
                    print(f"output_tokens length={len(output_tokens)}, i={i}")
        return output_tokens


class LlamaModel(Model):
    def __init__(self, name, revision, tokenizer_name=None, tokenizer_revision=None):
        # super().__init__(name, revision, tokenizer_name, tokenizer_revision)
        dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        self.model = LlamaForCausalLM.from_pretrained(name, revision=revision, torch_dtype=dtype,
                                                          trust_remote_code=True, device_map="auto" if torch.cuda.is_available() else None)
        # if torch.cuda.is_available():  # mostly for cpu debugging
        #     self.model = self.model.cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or name, revision=tokenizer_revision or revision,
                                                       padding_side="left", trust_remote_code=True)
        # self.tokenizer.pad_token = "<|endoftext|>"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id



def automodel_partial_arg_parser():
    """
    This is also used by peftmodel.py.
    """
    args = partial_arg_parser()
    args.add_argument("--name", type=str, required=True)
    args.add_argument("--revision", type=str)
    args.add_argument("--tokenizer_name", type=str)
    args.add_argument("--tokenizer_revision", type=str)
    args.add_argument("--name-override", type=str)
    args.add_argument("--load-adaptor", type=str, help="path to a pretrained adaptor")
    return args


def do_name_override(args):
    """
    Applies the --name-override flag, or uses the model name, correcting / and - which the rest of
    the toolchain does not like.
    """
    if args.name_override:
        name = args.name_override
    else:
        name = args.name.replace("/", "_").replace("-", "_")
    if args.load_adaptor:
        name += "_adaptor"
    return name


def main():
    args = automodel_partial_arg_parser()
    args = args.parse_args()
    if "llama" in args.name.lower(): # and (args.name.startswith("/") or args.name.startswith("./")):  # if local path provided
        model = LlamaModel(args.name, args.revision, args.tokenizer_name, args.tokenizer_revision)
    else:
        model = Model(args.name, args.revision, args.tokenizer_name, args.tokenizer_revision)
    if args.load_adaptor:
        model.merge_adaptor(args.load_adaptor)
    name = do_name_override(args)
    make_main(args, name, model.completions)


if __name__ == "__main__":
    main()