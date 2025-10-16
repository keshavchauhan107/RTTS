import logging

# Configure basic logging
logging.basicConfig(
    filename='logs/trade.log',          # Log file name
    filemode='w',                       # 'w' = overwrite on each run
    level=logging.INFO,                 # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create a logger instance
logger = logging.getLogger(__name__)
