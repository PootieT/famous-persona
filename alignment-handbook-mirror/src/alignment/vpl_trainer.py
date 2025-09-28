import pdb
from typing import Dict, List, Literal, Union, Optional, Any, Tuple
import os
import pickle

from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from alignment import OurDPOTrainer
from torch.utils.data import DataLoader
from trl.trainer.utils import DPODataCollatorWithPadding


class VPLModel(nn.Module):
    """
    variational preference learning encoder, takes in N pairs of preference data triples,
    return z (personal embedding) for each person
    """
    def __init__(self, hidden_dim: int, llm_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.llm_dim = llm_dim

        # paired encoder
        # TODO need to confirm these dimensions
        self.paired_enc = nn.Linear(2*llm_dim, hidden_dim)

        # self-attention layer
        self.q_emb = nn.Linear(hidden_dim, hidden_dim)
        self.k_emb = nn.Linear(hidden_dim, hidden_dim)
        self.v_emb = nn.Linear(hidden_dim, hidden_dim)
        self.seq_enc = nn.MultiheadAttention(hidden_dim, num_heads=1)

        # hidden to mu and sigma (https://medium.com/@rekalantar/variational-auto-encoder-vae-pytorch-tutorial-dce2d2fe0f5f)
        self.mean_layer = nn.Linear(hidden_dim, llm_dim)
        self.logvar_layer = nn.Linear(hidden_dim, llm_dim)

    def forward(self, chosen_embs: torch.FloatTensor, rejected_embs: torch.FloatTensor) -> torch.FloatTensor:
        """
        Given chosen and rejected data embeddings, pair encode them, pass through
        a sequence encoder (self-attention layer), generate mu and sigma, and sample
        z from mu and sigma

        Args:
            chosen_embs: (batch X llm_hidden_dim)
            rejected_embs: (batch X llm_hidden_dim)

        Returns:
            (llm_hidden_dim, )
        """
        # get paired embedding per pair of chosen/rejected data
        paired_embs = torch.cat([chosen_embs, rejected_embs], dim=1)
        paired_embs = self.paired_enc(paired_embs)

        # pass through self-attention layer with mean pooling
        q = self.q_emb(paired_embs)
        k = self.k_emb(paired_embs)
        v = self.v_emb(paired_embs)
        attn_out, attn_weights = self.seq_enc(q, k, v)
        pooled_out = attn_out.mean(dim=0)

        # convert to mu + sigma, then sample z
        mean, logvar = self.mean_layer(pooled_out), self.logvar_layer(pooled_out)
        epsilon = torch.randn_like(logvar).to(chosen_embs.device)
        z = mean + logvar * epsilon
        return z


class VPLTrainer(OurDPOTrainer):

    def __init__(
        self, **kwargs,
    ):
        """
        Our implementation of VPL (https://arxiv.org/pdf/2408.10075). Instead of
        training reward model, we learn a variational personal embedding (z) and
        prefix it to chosen/reject for prefixed DPO.
        Args:
            num_total_preference_pairs: (K in VPL paper)
            num_preference_pairs: (N in VPL paper)
            preference_encoder_hidden_dim: hidden size of variational encoder
            **kwargs:
        """
        super(VPLTrainer, self).__init__(**kwargs)

        self.num_total_preference_pairs = kwargs["model_args"].vpl_k
        self.num_preference_pairs = kwargs["model_args"].vpl_n
        self.preference_encoder_hidden_dim = kwargs["model_args"].vpl_hidden

        # define personal embedding VAE encoder
        self.vae_encoder = VPLModel(self.preference_encoder_hidden_dim, self.model.config.hidden_size)

        # if we are loading the model for continued training or eval
        self.weight_path = f"{kwargs['model_args'].model_name_or_path}/vpl_model.safetensors"
        if os.path.exists(self.weight_path):
            print(f"Found existing vpl weight file at {self.weight_path}, loading in safe tensor!")
            with safe_open(self.weight_path, framework="pt", device="cpu") as f:
                state_dict = {}
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
                self.vae_encoder.load_state_dict(state_dict)

        # forward pass and cache embeddings for (subset of) train split
        self.load_or_compute_and_cache_embeddings(kwargs)

        # if we want to cache eval persona's embedding during training, they need to be part
        # of training data. Once we compute cached embedding, remove those people from training
        # data for CV purpose
        if kwargs["model_args"].vpl_exclude_people is not None:
            print(f"Removing people from training loop and eval during training: "
                  f"{kwargs['model_args'].vpl_exclude_people}")
            # include proper and lower case names to catch all cases
            self.vpl_exclude_people_list = kwargs["model_args"].vpl_exclude_people.split(",")
            self.vpl_exclude_people_list.extend([name.lower().replace(" ", "") for name in self.vpl_exclude_people_list])
            train_len_before = len(self.train_dataset)
            self.train_dataset = self.train_dataset.filter(lambda row: row["name"] not in self.vpl_exclude_people_list)
            print(f"Training data filtered from {train_len_before} to {len(self.train_dataset)}")
            if isinstance(self.eval_dataset, dict):
                for name, eval_dataset in self.eval_dataset.items():
                    eval_len_before = len(eval_dataset)
                    self.eval_dataset[name] = eval_dataset.filter(lambda row: row["name"] not in self.vpl_exclude_people_list)
                    print(f"Eval data {name} filtered from {eval_len_before} to {len(self.eval_dataset[name])}")
            else:
                eval_len_before = len(self.eval_dataset)
                self.eval_dataset = self.eval_dataset.filter(lambda row: row["name"] not in self.vpl_exclude_people_list)
                print(f"Eval data filtered from {eval_len_before} to {len(self.eval_dataset)}")

    def load_or_compute_and_cache_embeddings(self, kwargs):
        cached_embedding_path = f"{self.args.output_dir}/cached_embedding.pkl"
        if kwargs["model_args"].vpl_load_cached_embedding and os.path.exists(cached_embedding_path):
            print(f"Embedding cache found in {cached_embedding_path}, loading ...")
            with open(cached_embedding_path, "rb") as f:
                cache = pickle.load(f)
            self.personal_train_data_indices = cache["personal_train_data_indices"]
            self.chosen_personal_train_emb = cache["chosen_personal_train_emb"]
            self.rejected_personal_train_emb = cache["rejected_personal_train_emb"]
        else:
            print(f"Embedding cache not found or vpl_load_cached_embedding=False, computing cache ...")
            self.compute_cached_embedding(kwargs["tokenizer"].pad_token_id, kwargs.get("label_pad_token_id", -100))
            cache = {
                "personal_train_data_indices": self.personal_train_data_indices,
                "chosen_personal_train_emb": self.chosen_personal_train_emb,
                "rejected_personal_train_emb": self.rejected_personal_train_emb
            }
            with open(cached_embedding_path, "wb") as f:
                pickle.dump(cache, f)

    def compute_cached_embedding(self, pad_token_id, label_pad_token_id):
        """
        To save computation, we cache the baseline model's embedding on the chosen/reject throughout
        training.
        """
        # subset part of train to obtain the z embeddings
        self.personal_train_data_indices = self.initialize_name_to_train_indices_map()
        all_personal_indices = []
        [all_personal_indices.extend(indices) for indices in self.personal_train_data_indices.values()]
        self.personal_emb_train_data = self.train_dataset.select(all_personal_indices)
        self.personal_train_data_indices = self.update_name_to_train_indices_map()

        # now actually forward the data and get last token embedding
        collator = DPODataCollatorWithPadding(
            pad_token_id=pad_token_id,
            label_pad_token_id=label_pad_token_id,
            is_encoder_decoder=self.is_encoder_decoder,
        )
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for field in ["chosen", "rejected"]:
                    print(f"Computing and cacheing {field} embedding for personal token generation...")
                    data_loader = DataLoader(
                        self.personal_emb_train_data,  # type: ignore
                        shuffle=False,
                        collate_fn=collator,
                        batch_size=self.args.per_device_eval_batch_size,
                    )
                    all_pooled_output = []
                    for batch in tqdm(data_loader, total=len(self.personal_emb_train_data)//self.args.per_device_eval_batch_size):
                        # forward model to get embedding of full prompt + chosen/rejected
                        output = self.model(
                            batch[f"{field}_input_ids"].to(self.model.device),
                            attention_mask=batch[f"{field}_attention_mask"].to(self.model.device),
                            return_dict=True,
                            output_hidden_states=True
                        )

                        # using last token embedding as pooling strategy (following VPL)
                        last_token_idx = batch[f"{field}_attention_mask"].sum(1) - 1
                        # at each datapoint, fetch the last token embedding
                        pooled_output = output["hidden_states"][-1].cpu()[torch.arange(len(last_token_idx)), last_token_idx]
                        all_pooled_output.append(pooled_output)
                    setattr(self, f"{field}_personal_train_emb", torch.cat(all_pooled_output))

    def update_name_to_train_indices_map(self) -> Dict[str, List[str]]:
        """
        previous indices are mapped to full training data, embeddings are calculated only on
        subset of that, so now we change the mapped indices to the cached embeddings
        Returns:
            dictionary of name mapped to indices of datapoints in cached embeddings
        """
        old_indices_span = {}
        combined_indices = []
        for name, indices in self.personal_train_data_indices.items():
            old_indices_span[name] = (len(combined_indices), len(combined_indices)+len(indices))
            combined_indices.extend(indices)

        new_indices = np.argsort(np.argsort(combined_indices))
        new_map = {}
        for name in self.personal_train_data_indices.keys():
            new_map[name] = new_indices[old_indices_span[name][0]: old_indices_span[name][1]]
        return new_map

    def initialize_name_to_train_indices_map(self) -> Dict[str, List[int]]:
        """
        Given training data, we want to find name to all their data point indices mapping
        Returns:
            dictionary of name mapped to indices of datapoints
        """
        m = {}
        for i, row in enumerate(self.train_dataset):
            if row["name"] not in m:
                m[row["name"]] = [i]
            else:
                m[row["name"]].append(i)
        m_subset = {}
        for name, indices in m.items():
            m_subset[name] = np.random.choice(indices, size=min(self.num_total_preference_pairs, len(indices)), replace=False)
        return m_subset

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Instead of forward passing starting from token ids, we are going to convert them
        to embeddings, concatenate with persona prefix embedding, then forward to model
        as embedding as input.
        Args:
            model: model we are finetuning
            batch: contains chosen/rejected fields, and name as well

        Returns:

        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )

        # Here we modify concatenated_batch to include personal embedding (z) from
        # forwarding the VAE encoder
        personal_emb = self.get_personal_embedding(batch["name"])
        concatenated_batch = self.prefix_with_personal_embedding(personal_emb, concatenated_batch)

        all_logits = model(
            # concatenated_batch["concatenated_input_ids"],
            # depending on model some (llama) call it input_embs, some (mistral) call it inputs_embeds
            inputs_embeds=concatenated_batch["concatenated_input_embs"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        ).logits

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=self.loss_type == "ipo",
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def get_personal_embedding(self, names: List[str]) -> torch.FloatTensor:
        """
        For each person in the batch, randomly select N personal training data
        pass through VAE encoder to obtain z embedding.

        Returns:
            personal embedding of size (batch_size, emb_dim)
        """
        # get the name to indices mapping to get index of training data for the people
        personal_indices_list = [self.personal_train_data_indices[n] for n in names]
        personal_emb_batch = []

        # for each person in the batch, get the personal embedding
        for personal_indices in personal_indices_list:
            indices = np.random.choice(personal_indices, min(self.num_preference_pairs, len(personal_indices)), replace=False)
            chosen_train_embs = self.chosen_personal_train_emb[indices]
            rejected_train_embs = self.rejected_personal_train_emb[indices]
            vae_emb = self.vae_encoder(chosen_train_embs, rejected_train_embs)
            personal_emb_batch.append(vae_emb)

        personal_emb_batch = torch.stack(personal_emb_batch)
        return personal_emb_batch

    def prefix_with_personal_embedding(
        self, personal_emb: torch.FloatTensor, concatenated_batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Dict[str, Union[List, torch.FloatTensor, torch.LongTensor]]:
        """
        Replace input_ids with model's token embeddings, then prefix the embedding
        tensor with personal_emb in front of chosen and rejected location
        Args:
            personal_emb: torch float tensor of size (batch_size, emb_dim)
            concatenated_batch: Dictionary containing input_ids, masked_tokens, labels
                for combined chosen/rejected string, each with size (batch_size * 2, emb_dim)

        Returns:
            batch with additional field 'concatenated_input_embs'
        """
        device = self.model.device
        # shape of input_embs is (bs*2, seq_len, emb_dim)
        input_embs = self.model.get_input_embeddings()(concatenated_batch["concatenated_input_ids"])
        # TODO maybe insert at beginning of instruction and modify all fields, but this should do for now,
        personal_emb_cat = torch.cat([personal_emb, personal_emb]).reshape(input_embs.shape[0], 1, -1).to(device)
        concatenated_batch["concatenated_input_embs"] = torch.cat([personal_emb_cat, input_embs], dim=1)
        concatenated_batch["concatenated_attention_mask"] = torch.cat([
            torch.ones(input_embs.shape[0], 1).to(device),
            concatenated_batch["concatenated_attention_mask"]], dim=1)
        concatenated_batch["concatenated_labels"] = torch.cat([
            (torch.ones(input_embs.shape[0], 1)*self.label_pad_token_id).long().to(device),
            concatenated_batch["concatenated_labels"]], dim=1)
        return concatenated_batch

    def save_model(self, output_dir: str, **kwargs):
        super(OurDPOTrainer, self).save_model(output_dir, **kwargs)
        out_weight_path = f"{output_dir}/vpl_model.safetensors"
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        save_file(self.vae_encoder.state_dict(), out_weight_path)

