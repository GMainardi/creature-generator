from tokenizers import ByteLevelBPETokenizer
import torch

class Tokenizer:

    def __init__(self, dataset: str) -> None:
        self.tokenizer = ByteLevelBPETokenizer()

        self.tokenizer.train(files=[dataset], vocab_size=50_000, min_frequency=2, special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ])
    
    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def tokenize(self, dataset: str) -> torch.Tensor:
        return torch.tensor(self.tokenizer.encode(dataset).ids, dtype=torch.long)


    def detokenize(self, dataset: torch.Tensor) -> str:
        return self.tokenizer.decode(dataset.view(-1).tolist())
    