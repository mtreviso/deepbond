import re


class Cleaner:
    def __init__(self):
        """Static class with useful methods for sanitize a given text"""
        pass

    @staticmethod
    def lowercase(text):
        return text.lower()

    @staticmethod
    def trim(text):
        return re.sub(r' +', ' ', text).strip()

    @staticmethod
    def transform_numbers(text):
        return re.sub(r'\d', '0', text)

    @staticmethod
    def transform_decimals(text):
        return re.sub(r'\d+[\.\,]\d+', '<DECIMAL>', text)

    @staticmethod
    def transform_urls(text):
        return re.sub(r'(http|https)://[^\s]+', '<URL>', text)

    @staticmethod
    def transform_emails(text):
        return re.sub(r'[^\s]+@[^\s]+', '<EMAIL>', text)

    @staticmethod
    def transform_dollar(text):
        return re.sub(r'\d+[\.\,]?[0.9]*\ ?[a-zA-Z]*[\$\£\€]', '<MOEDA>', text)

    @staticmethod
    def transform_hours(text):
        text = re.sub(r'(\d+([hms]\d*)+|\d+([\:\-]\d*)[hms])+', '<HORA>', text)
        return re.sub(r'HOUR([\:\-]?HOUR)*', '<HORA>', text)

    @staticmethod
    def transform_dates(text):
        return re.sub(r'\d+[\/\-\:]\d+([\/\-\:]\d+)?', '<DATA>', text)

    @staticmethod
    def fix_quotes(word):
        """
        from: https://github.com/erickrf/nlpnet/blob/master/nlpnet/utils.py
        """
        word = re.sub(r"(?u)(^|\W)[‘’′`']", r'\1"', word)
        word = re.sub(r"(?u)[‘’`′'](\W|$)", r'"\1', word)
        word = re.sub(r'(?u)[‘’`′“”]', '"', word)
        return word

    @staticmethod
    def fix_mistyped_tokens(word):
        """
        adpated from:
        https://github.com/erickrf/nlpnet/blob/master/nlpnet/utils.py
        """
        # take care with ellipses
        word = re.sub(r'(?<!\.)\.\.(?!\.)', '.', word)
        # replaces special ellipsis character
        word = word.replace('…', '...')
        return word
