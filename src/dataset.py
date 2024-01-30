from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch
from omegaconf import DictConfig


class CodingDataset(Dataset):
    """
    A Dataset class for loading and tokenizing coding data.
    """

    def __init__(self, tokenizer, cfg: DictConfig):
        """
        Initialize the CodingDataset.

        Args:
            tokenizer (AutoTokenizer): Tokenizer for processing the text.
            cfg (DictConfig): The configuration object from Hydra.
        """
        self.dataset = load_dataset("code_search_net", "python", split='train', download_mode="reuse_dataset_if_exists")
        self.tokenizer = tokenizer
        self.max_length = cfg.model.max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        code = data['code'] if 'code' in data else ''
        docstring = data['docstring'] if 'docstring' in data else ''

        try:
            encoding = self.tokenizer(
                code,
                docstring,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return encoding.input_ids.squeeze(0), encoding.attention_mask.squeeze(0)
        except Exception as e:
            print(f"Error tokenizing data at index {idx}: {e}")
            return torch.tensor([]), torch.tensor([])
