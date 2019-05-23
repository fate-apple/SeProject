#encoding:utf-8
import os
import logging
from pathlib import Path
from logging import Logger
from logging.handlers import TimedRotatingFileHandler

def init_logger(log_name,log_dir):
    print(os.getcwd())
    if not isinstance(log_dir,Path):
        log_dir = Path(log_dir)
    if not log_dir.exists():
        #忽略 FileExistsError 异常:如果目录已存在，则不会引起错误。
        print('mkdir :  {}'.format(log_dir))
        log_dir.mkdir(exist_ok=True)
    #from logging import Logger
    if log_name not in Logger.manager.loggerDict:
        logger = logging.getLogger(log_name)
        logger.setLevel(logging.DEBUG)

        #from logging.handlers import TimedRotatingFileHandler
        handler = TimedRotatingFileHandler(filename=str(log_dir / f"{log_name}.log"),)
        datefmt = '%Y-%m-%d %H:%M:%S'
        format_str = '[%(asctime)s]: %(name)s %(filename)s[line:%(lineno)s] %(levelname)s %(message)s'
        formatter = logging.Formatter(format_str,datefmt)
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)

        handler =TimedRotatingFileHandler(filename=str(log_dir / "ERROR.log"),when="D",backupCount=30)
        handler.setFormatter(formatter)
        handler.setLevel(logging.ERROR)
        logger.addHandler(handler)
    logger = logging.getLogger(log_name)
    return logger


#debug
init_logger('log_name','log_dir')


