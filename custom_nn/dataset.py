import torch

class Dataset():

    def __init__(self, filename: str, block_size: int = 8, batch_size: int = 4) -> None:

        self.filename = filename

        with open(self.filename, 'r', encoding='utf-8') as file:
            self.data = file.read()        
    

    def train_test_split(self, test_size: float = 0.1) -> None:

        samples = self.data.split('\n\n')[:-1]

        split_point = int(len(samples)*(1-test_size))

        train = ''.join(samples[:split_point])
        test = ''.join(samples[split_point:])

        return train, test

    @staticmethod
    def get_batch(
        dataset: torch.tensor, 
        block_size: int = 8, 
        batch_size: int = 4
    ) -> tuple[torch.stack, torch.stack]:
                
        ix = torch.randint(0, len(dataset) - block_size, (batch_size,))

        batch_x = torch.stack([dataset[i:i+block_size] for i in ix])
        batch_y = torch.stack([dataset[i+1:i+block_size+1] for i in ix])

        return batch_x, batch_y


        