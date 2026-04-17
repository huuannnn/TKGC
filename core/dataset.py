from torch.utils.data import Dataset
import os
import pickle
import torch
import utils


class TKGDataset(Dataset):
    def __init__(self, dataset: str = "YAGO"):
        super().__init__()
        self.dataset_name = dataset
        self.data_path = os.path.join('./data', dataset)
        
        self._load_statistics()
        self._load_split_data('train')
        self.has_valid = self._load_split_data('dev', required=False)
        self._load_split_data('test')
        self._load_total_data()

    def _load_statistics(self):
        """Load dataset statistics."""
        self.num_nodes, self.num_rels, self.num_t = utils.get_total_number(
            self.data_path, 'stat.txt'
        )

    def _load_file(self, filename: str):
        """Load data from a pickle or torch file."""
        filepath = os.path.join(self.data_path, filename)
        with open(filepath, 'rb') as f:
            if filename.endswith('frequency.txt') and self.dataset_name == 'GDELT':
                return torch.load(f).toarray()
            data = pickle.load(f)
            if hasattr(data, 'toarray'):
                return data.toarray()
            return data

    def _load_split_data(self, split: str, required: bool = True) -> bool:
        """Load data for a split (train/dev/test).
        
        Args:
            split: Data split name ('train', 'dev', 'test')
            required: If False, gracefully handle missing files
            
        Returns:
            True if data loaded successfully, False otherwise
        """
        try:
            # Load quadruples
            quadruple_file = f'{split}.txt' if split != 'dev' else 'valid.txt'
            data, _ = utils.load_quadruples(self.data_path, quadruple_file)
            setattr(self, f'{split}_data', data)
            
            # Load history data (subject and object)
            for abbr, entity_type in [('sub', 's'), ('ob', 'o')]:
                history_list = self._load_file(f'{split}_history_{abbr}.txt')
                setattr(self, f'{split}_{entity_type}_history', history_list[0])
                setattr(self, f'{split}_{entity_type}_history_t', history_list[1])
            
            # Load labels
            for entity_type in ['s', 'o']:
                label_data = self._load_file(f'{split}_{entity_type}_label.txt')
                setattr(self, f'{split}_{entity_type}_label', label_data)
            
            # Load frequencies
            for entity_type in ['s', 'o']:
                frequency_data = self._load_file(f'{split}_{entity_type}_frequency.txt')
                setattr(self, f'{split}_{entity_type}_frequency', frequency_data)
            
            return True
            
        except (FileNotFoundError, Exception) as e:
            if required:
                raise
            print(f"{self.dataset_name} does not have {split} set: {e}")
            # Initialize as None
            for attr in ['data', 's_history', 's_history_t', 'o_history', 'o_history_t',
                        's_label', 'o_label', 's_frequency', 'o_frequency']:
                setattr(self, f'{split}_{attr}', None)
            return False

    def _load_total_data(self):
        """Load combined training data."""
        try:
            splits = ['train.txt', 'valid.txt', 'test.txt'] if self.has_valid else ['train.txt', 'test.txt']
            self.total_data, _ = utils.load_quadruples(self.data_path, *splits)
        except Exception as e:
            print(f"Error loading total data for {self.dataset_name}: {e}")
            self.total_data = self.train_data
        
        self.data = self.total_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


class TKGDataLoader:
    """Custom DataLoader for TKG dataset with complex batch structure."""
    
    def __init__(self, quadruples, s_history, o_history, s_label, o_label, 
                 s_frequency, o_frequency, batch_size):
        """Initialize TKG DataLoader.
        
        Args:
            quadruples: Array of quadruples (s, r, o, t)
            s_history: Subject history data
            o_history: Object history data
            s_label: Subject labels
            o_label: Object labels
            s_frequency: Subject frequencies
            o_frequency: Object frequencies
            batch_size: Batch size
        """
        self.quadruples = quadruples
        self.s_history = s_history
        self.o_history = o_history
        self.s_label = s_label
        self.o_label = o_label
        self.s_frequency = s_frequency
        self.o_frequency = o_frequency
        self.batch_size = batch_size
        self.num_samples = len(quadruples)
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
    
    def __iter__(self):
        """Iterate over batches."""
        for i in range(0, self.num_samples, self.batch_size):
            end_idx = min(i + self.batch_size, self.num_samples)
            yield [
                self.quadruples[i:end_idx],
                self.s_history[i:end_idx],
                self.o_history[i:end_idx],
                self.s_label[i:end_idx],
                self.o_label[i:end_idx],
                self.s_frequency[i:end_idx],
                self.o_frequency[i:end_idx]
            ]
    
    def __len__(self):
        return self.num_batches
