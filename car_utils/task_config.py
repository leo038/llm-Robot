import logging

####### 日志设置  ##################################
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s',
                    # filename='run_demo_20240619.log',
                    filemode='a')

global_logger = logging.getLogger(__name__)

#################################################
