from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset


class CodingDataset(Dataset):
    def __init__(self, tokenizer, split='train', max_length=512):
        """
        Initialize the CodingDataset.

        Parameters:
        tokenizer (AutoTokenizer): Tokenizer for processing the text.
        split (str): The specific split of the dataset to use (e.g., 'train').
        max_length (int): Maximum length of the tokenized strings.
        """
        self.dataset = load_dataset("code_search_net", "python", split=split, download_mode="reuse_dataset_if_exists")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        code = data['code'] if 'code' in data else ''
        docstring = data['docstring'] if 'docstring' in data else ''

        # Tokenize code and docstring
        encoding = self.tokenizer(
            code,
            docstring,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoding.input_ids, encoding.attention_mask