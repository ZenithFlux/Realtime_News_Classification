import logging
from datetime import datetime as dt
import os
import sys

if not os.path.exists("logs"):
    os.mkdir("logs")

logging.basicConfig(filename = f"logs/{dt.now().strftime('%Y-%m-%d_%H-%M')}.log",
                    format= "%(levelname)s::%(filename)s:%(funcName)s:: %(message)s\n",
                    level="INFO")


if __name__ == "__main__":
    logging.info("Logger working!")
    logging.info("Logger working!")