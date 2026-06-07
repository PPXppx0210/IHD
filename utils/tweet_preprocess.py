import re
import string
import emoji

def remove_non_ascii_chars(text):
    ascii_chars = set(string.printable)
    for c in text:
        if c not in ascii_chars:
            text = text.replace(c,'')
    return text

def fix_ax_nots(text):
    text = text.replace(" dont ", " do not ")
    text = text.replace(" don't ", " do not ")
    text = text.replace(" doesnt ", " does not ")
    text = text.replace(" doesn't ", " does not ")
    text = text.replace(" didnt ", " did not ")
    text = text.replace(" didn't ", " did not ")
    text = text.replace(" wont ", " will not ")
    text = text.replace(" won't ", " will not ")
    text = text.replace(" cant ", " cannot ")
    text = text.replace(" can't ", " cannot ")
    text = text.replace(" couldnt ", " could not ")
    text = text.replace(" couldn't ", " could not ")
    text = text.replace(" shouldnt ", " should not ")
    text = text.replace(" shouldn't ", " should not ")
    text = text.replace(" wouldnt ", " would not ")
    text = text.replace(" wouldn't ", " would not ")
    text = text.replace(" mustnt ", " must not ")
    text = text.replace(" mustn't ", " must not ")
    return text

def fix_personal_pronouns_and_verb(text):
    text = text.replace(" im ", " i am ")
    text = text.replace(" youre ", " you are ")
    text = text.replace(" hes ", " he is ") # ? he's can be he has as well
    text = text.replace(" shes ", " she is ")
    # we are -> we're -> were  ---- were is a valid word
    text = text.replace(" theyre ", " they are ")
    text = text.replace(" ive ", " i have ")
    text = text.replace(" youve ", " you have ")
    text = text.replace(" weve ", " we have ")
    text = text.replace(" theyve ", " they have ")
    text = text.replace(" youll ", " you will ")
    text = text.replace(" theyll ", " they will ")
    return text

def fix_special_chars(text):
    text = text.replace("&amp;", " and ")
    # text = text.replace("--&gt;", "")
    return text

def preprocess_text_forbert(text: str) -> str:
    text = re.sub(r'RT @(\w+):', '_retweet_', text) # remove retexts
    text = re.sub(r'@[\w_]+', '@user', text) # remove at-links
    text = re.sub(r'http(s?)://[a-zA-Z0-9.?/&=:]+', '_httpurl_', text) # remove urls
    text = re.sub(r'\n', '', text) # remove new lines
    text = emoji.demojize(text, delimiters=(' _','_ ')) # demojize emojis, i.e. _broken_heart_

    text = remove_non_ascii_chars(text)
    text = fix_ax_nots(text)
    text = fix_personal_pronouns_and_verb(text)
    text = fix_special_chars(text)
    text = re.sub(r' +', ' ', text) # 去除多余的空格
    return text.strip().lower() # lowercase
