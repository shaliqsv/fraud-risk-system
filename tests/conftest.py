from dotenv import load_dotenv

# Must run before any `api.*` module is imported, so DATABASE_URL etc. are in
# os.environ before api/prediction_logger.py reads them at import time.
load_dotenv()
