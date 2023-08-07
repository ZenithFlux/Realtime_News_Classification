import logging
from datetime import datetime as dt
import os

os.makedirs("logs", exist_ok=True)

logging.basicConfig(filename = f"logs/{dt.now().strftime('%Y-%m-%d_%H-%M')}.log",
                    format= "%(levelname)s::%(filename)s:%(funcName)s:: %(message)s\n",
                    level="INFO")


if __name__ == "__main__":
    logging.info("Logger working!")
    logging.info("Logger working!")