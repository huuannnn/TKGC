import logging
import os


class Logger:
    """Custom logger that writes to both console and .log file.
    
    Creates a folder structure: version_<id>_dataset/
    with training.log file inside, where id is auto-incremented
    """
    
    def __init__(self, save_dir, dataset_name, version='version'):
        """
        Initialize logger with auto-incrementing ID and create versioned folder.
        
        Args:
            save_dir: Directory where version_<id>_dataset folder will be created
            dataset_name: Name of the dataset
            version: Version string (default: 'version')
        
        Returns:
            log_dir: Path to the created version_<id>_dataset folder
        """
        # Find the next available ID
        log_id = self._get_next_id(save_dir, version, dataset_name)
        
        # Create folder name: version_<id>_dataset
        folder_name = f"{version}_{log_id}_{dataset_name}"
        self.log_dir = os.path.join(save_dir, folder_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create log file inside the folder
        self.log_file = os.path.join(self.log_dir, 'training.log')
        
        # Setup logger
        self.logger = logging.getLogger(self.log_file)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers to avoid duplicate logs
        self.logger.handlers = []
        
        # File handler
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        self.logger.addHandler(fh)
    
    def get_log_dir(self):
        """Return the path to the versioned log directory."""
        return self.log_dir
    
    def _get_next_id(self, log_dir, version, dataset_name):
        """Find the next available ID by checking existing log folders."""
        prefix = f"{version}_"
        suffix = f"_{dataset_name}"
        
        max_id = 0
        if os.path.exists(log_dir):
            for foldername in os.listdir(log_dir):
                if foldername.startswith(prefix) and foldername.endswith(suffix):
                    # Extract ID from folder name: version_id_dataset
                    try:
                        id_str = foldername[len(prefix):-len(suffix)]
                        id_num = int(id_str)
                        max_id = max(max_id, id_num)
                    except (ValueError, IndexError):
                        pass
        
        return max_id + 1
    
    def write(self, message):
        """Write message to both console and log file."""
        print(message)
        self.logger.info(message)
    
    def close(self):
        """Close all handlers."""
        for handler in self.logger.handlers:
            handler.close()
