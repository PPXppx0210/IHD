import datetime
import logging

# logging config
def Beijing_TimeZone_Converter(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp
# logging.Formatter.converter = time.gmtime
logging.Formatter.converter = Beijing_TimeZone_Converter