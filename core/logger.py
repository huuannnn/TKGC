import logging


class Logger:
    """Custom logger that writes to both console and .log file."""
    
    def __init__(self, log_file):
        self.log_file = log_file
        self.logger = logging.getLogger(log_file)
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        self.logger.addHandler(fh)
    
    def write(self, message):
        print(message)
        self.logger.info(message)
    
    def close(self):
        for handler in self.logger.handlers:
            handler.close()
