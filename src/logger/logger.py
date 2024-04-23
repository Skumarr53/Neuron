import os, logging
from datetime import datetime


LOGFILE = f"logs/{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"

os.makedirs(os.path.dirname(LOGFILE), exist_ok=True)


logging.basicConfig(
  filename=LOGFILE,
  filemode="a",
  format="[%(asctime)s] - %(name)s - %(levelname)s - %(message)s",
  datefmt="%y-%m-%d %H:%M:%S",
  level=logging.INFO,
)


if __name__ == "__main__":
    logging.info("Hello, World!")