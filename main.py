from config import config
from preprocessing import preprocess_projections
from reconstruction import reconstruct_sinograms
from utilities import setup_logging

def main():
    """
    Main workflow for preprocessing and reconstructing projections.
    """
    # Step 1: Setup logging
    logger = setup_logging(config["log_path"])

    # Step 2: Preprocess projections
    logger.info("Starting preprocessing...")
    preprocess_projections(config, logger)

    # Step 3: Reconstruct sinograms
    logger.info("Starting reconstruction...")
    reconstruct_sinograms(config, logger)

if __name__ == "__main__":
    main()
