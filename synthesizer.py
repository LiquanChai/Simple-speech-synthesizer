import os
import sys
import SimpleAudio as SA
import argparse
import numpy as np
import re
import nltk
from datetime import datetime

### NOTE: DO NOT CHANGE ANY OF THE EXISITING ARGUMENTS
parser = argparse.ArgumentParser(
    description='A basic text-to-speech app that synthesises an input phrase using monophone unit selection.')
parser.add_argument('--monophones', default="monophones", help="Folder containing monophone wavs")
parser.add_argument('--play', '-p', action="store_true", default=False, help="Play the output audio")
parser.add_argument('--outfile', '-o', action="store", dest="outfile", type=str, help="Save the output audio to a file",
                    default=None)
parser.add_argument('phrase', nargs=1, help="The phrase to be synthesised")

# Arguments for extensions
parser.add_argument('--spell', '-s', action="store_true", default=False,
                    help="Spell the phrase instead of pronouncing it")
parser.add_argument('--volume', '-v', default=None, type=float,
                    help="A float between 0.0 and 1.0 representing the desired volume")

args = parser.parse_args()

############################################################################
# Section for regular expressions                                          #
# regular expressions is used in text tokenization, and text normalization #
############################################################################
# expression used for text tokenization
word_only_pattern = r"[A-Za-z-]+" # basic implementation, tokenize all words from the input and ignore others

# extended implementation, tokenize words, punctuations, numbers and dates
date_number_punctuation_pattern = r"""
             (?x)           # set flag to allow verbose regexps
             \d?\d/\d?\d(?:/(?:\d\d)?\d\d)?     # date, (D)D/(M)M(/(YY)YY)
             |\d+(?:\.\d+)? # number, including digits, it must order before the expression for word and punctuation
             |\w+           # word
             |[!?.,]        # punctuations, only [!?.,] will be recognized
          """
# this regular expression build according to:
# http://stackoverflow.com/questions/22175923/nltk-regexp-tokenizer-not-playing-nice-with-decimal-point-in-regex
# http://stackoverflow.com/questions/36353125/nltk-regular-expression-tokenizer
# https://github.com/nltk/nltk/issues/1206#issuecomment-156470847

# expression used for text normalization
date_pattern = r"\d?\d/\d?\d(?:/(?:\d\d)?\d\d)?"
number_pattern = r"\d+(?:\.\d+)?"

# print args # for testing purpose
print args.monophones


###############################
# Section for audio synthesis #
###############################
"""
Synthesis class
Object in the class is used for processing speech
given the word/letter pronunciation sequence, generate corresponding audio data
"""
class Synth(object):
    def __init__(self, wav_folder, rate, sp_time=250, lp_time=500):
        self.phones = {} # phones used for synthesis (key: phone name, value:audio object)
        self.rate = rate        # synthesis rate, equal to the rate of the pronunciation files used
        self.sp_time = sp_time  # set time for short pause for speech
        self.lp_time = lp_time  # set time for long pause for speech
        self.get_wavs(wav_folder) # from the files given, load the audio files, and store in the phone dictionary.
                                  # It should be in the last, orders do matter
    # function that load all audio data (all possible pronunciation audio files) into the synthesis object
    # input: path of wav_floder (string), outpur: non empty self.phones attribute
    def get_wavs(self, wav_folder):
        for root, dirs, files in os.walk(wav_folder, topdown=False): # loading phonemes from file
            for file in files: # file names -> str
                phone_name = file.split('.')[0]
                self.phones[phone_name] = SA.Audio() # each phone name as each phone object
                self.phones[phone_name].load(os.path.join(wav_folder, file))
        # load for short pause and long pause, and create data (salience) (for punctuation)
        # reference: add echo method in SimpleAudio
        self.phones["sp"] = SA.Audio(rate=self.rate)
        self.phones["sp"].data = np.zeros(self.sample_converter(self.sp_time), self.phones["sp"].nptype)
        self.phones["lp"] = SA.Audio(rate=self.rate)
        self.phones["lp"].data = np.zeros(self.sample_converter(self.lp_time), self.phones["lp"].nptype)

    # concatenate a audio data sequence into a single output data
    # input: list of audio data (i.e. a list of numpy array), output: a single audio data (i.e. a single numpy array)
    # reference:
    # http://stackoverflow.com/questions/9236926/concatenating-two-one-dimensional-numpy-arrays
    def concatenate(self, phone_seq):
        data_sequnce = []
        for phone in phone_seq:
            data_sequnce.append(self.phones[phone].data)
        return np.concatenate(data_sequnce)

    # a method that convert time(ms) into the number samples
    # input: time in illisecond (int), output: number of samples (int)
    def sample_converter(self, time):  # time in milliseconds, rate s^-1
        return int((time / 1000.0) * self.rate)


#####################################################################################################
# Section for language processing,                                                                  #
# it contains 4 classes:                                                                            #
# Word_to_phone_seq_generator, Letter_to_phone_seq_generator, Number_normalizer and Date_normalizer #
#####################################################################################################
# function check if -s appear in the command line or not
# distinguish whether the pronunciation is spelling or word pronunciation
def get_phone_seq(phrase):
    if args.spell:
        return Letter_to_phone_sequence_generator(Word_to_phone_seq_generator(phrase).word_tokens).letter_phone_seq
    else:
        return Word_to_phone_seq_generator(phrase).word_phone_seq

"""
Words to phone sequence class
object in the class is used to generate word pronunciation sequence
given the corresponding phrase (in string form) (including words, numbers, dates, punctuations)
working process: phrase normalization -> word tokens sequence to phone sequence -> normalize the phone sequence
"""
class Word_to_phone_seq_generator():
    def __init__(self, phrase):
        self.input_phrase = phrase      # the input phrase for phone sequence generation (string)
        self.word_tokens = self.normalize_text(self.input_phrase) # normalized word tokens sequence (list of words)
        self.word_phone_seq = self.word_tokens_to_phone_seq(self.word_tokens) # word pronunciation sequence (list of pronunciations of the word tokens sequence)
                                                                              # the pronunciation sequence is recognizable by Synth object

    # function that normalize a phrase (including number, punctuation, number, date) into list of words
    # input: phrase (string), output: list of words (list of string)
    def normalize_text(self, phrase):  # phrase to a sequnce of normalized tokens
        r1 = re.compile(date_number_punctuation_pattern)  # extension version, standard version use word_only_pattern
        tokens = nltk.tokenize.regexp_tokenize(phrase, r1) # extract tokens (words, dates, numbers, punctuations)
        lower_case_tokens = [x.lower() for x in tokens]

        normalized_tokens = []
        r_date = re.compile(date_pattern)
        r_number = re.compile(number_pattern)
        r_word = re.compile(word_only_pattern)

        for token in lower_case_tokens:
            if r_date.match(token):
                word_date_tokens = nltk.tokenize.regexp_tokenize(DateNormalizer(token).normalized_word_tokens, r_word)
                for word in word_date_tokens:
                    normalized_tokens.append(word)
            elif r_number.match(token):
                word_number_tokens = nltk.tokenize.regexp_tokenize(Number_Normalizer(token).normalized_word_tokens, r_word)
                for word in word_number_tokens:
                    normalized_tokens.append(word)
            else:
                normalized_tokens.append(token)
        return normalized_tokens

    # function that produce the phone sequence of a given word token sequence
    # input: list of words (including punctuation) (list of string), output: list of pronunciation (list of string, items in list should be keys in Synth.phones)
    def word_tokens_to_phone_seq(self, tokens): # word tokens to phone sequence
        arpabet = nltk.corpus.cmudict.dict()
        phone_sequence = [] # sequence of phones returned
        for token in tokens:
            try:
                phone_sequence.append(arpabet[token][0]) # get the pronunciation sequence of the word token
            except KeyError: # not an recognizable word token
                if token in ",.?!":  # if the token is punctuation
                    if token == ",":
                        phone_sequence.append(["sp"]) # append short salience
                    else:
                        phone_sequence.append(["lp"]) # append long salience
                else:
                    print "the word %s can not be find in the cmudict, exit the program" % token
                    sys.exit()
        return self.normalise_phone_seq(phone_sequence)

    # function that convert pronunciation sequence in cmudict format (list of list of phones, separate by word)
    # into sequence of keys in Synth.phones
    # by flatten the list and stripping out the stress
    # input: list of word phone sequence (list of list of string), output: list of phone sequence (list of string)
    def normalise_phone_seq(self, phone_sequence):
        flat_phone_sequence = [phone.lower() for sublist in phone_sequence for phone in sublist]  # reference: http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
        strip_stress_sequence = ["".join([letter for letter in phone if not letter.isdigit()]) for phone in flat_phone_sequence]  # reference: http://stackoverflow.com/questions/12851791/removing-numbers-from-string
        return strip_stress_sequence

"""
Letter to phone sequence class
object in the class is used to generate letter pronunciation sequence
given the corresponding word token sequence (with punctuations)
working process: words to letters -> letter sequences to phone sequence -> normalize the phone sequence
"""
class Letter_to_phone_sequence_generator():
    def __init__(self, word_tokens):
        self.input_word_tokens = word_tokens    # the input word token sequence for generating letter token sequence
        self.letter_tokens = self.word_to_letter_seq(self.input_word_tokens)    # corresponding letter tokens sequence
        self.letter_phone_seq = self.letter_tokens_to_phone_seq(self.letter_tokens) # corresponding phone sequence for letter tokens sequence

    # function that convert sequence of word tokens into sequence of letter sequence
    # input: word tokens sequence (list of string),
    # output: letter tokens sequence (list of string, each item in list is a letter)
    def word_to_letter_seq(self, word_tokens):
        return [letter for word in word_tokens for letter in word]

    # function that produce the phone sequence of a given letter token sequence
    # input: list of letter (including punctuation) (list of string), output: list of pronunciation (list of string, items in list should be keys in Synth.phones)
    def letter_tokens_to_phone_seq(self, letter_tokens):
        arpabet = nltk.corpus.cmudict.dict()
        phone_sequence = []
        for letter in letter_tokens:
            try:
                if letter == "a": # "a" should be pronounced as "ey" instead of "ah"
                    phone_sequence.append(arpabet[letter][1])
                else:
                    phone_sequence.append(arpabet[letter][0])
            except KeyError:
                if letter in ",.?!": # if the token is punctuation
                    if letter == ",":
                        phone_sequence.append(["sp"]) # append a short pause
                    else:
                        phone_sequence.append(["lp"]) # append a long pause
                else:
                    print "the letter %s can not be find in the cmudict, exit the program" % letter
                    sys.exit()
        return self.normalise_phone_seq(phone_sequence)

    # function that convert pronunciation sequence in cmudict format (list of list of phones, separate by letter)
    # into sequence of keys in Synth.phones
    # by flatten the list and stripping out the stress
    # input: list of letter phone sequence (list of list of string), output: list of phone sequence (list of string)
    def normalise_phone_seq(self, phone_sequence):
        flat_phone_sequence = [phone.lower() for sublist in phone_sequence for phone in sublist]  # reference: http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
        strip_stress_sequence = ["".join([letter for letter in phone if not letter.isdigit()]) for phone in flat_phone_sequence]  # reference: http://stackoverflow.com/questions/12851791/removing-numbers-from-string
        return strip_stress_sequence


"""
Number normalization class
object in the class is used to convert numerical number (integar or decimal) (string)
into its corresponding word form (string)
working process: break numbers into integer part and decimal part
and convert each part from numbers to words
"""
class Number_Normalizer:
    def __init__(self, number_token_str):
        self.number_in_str = number_token_str # input string in numerical format
        self.normalized_word_tokens = self.number_normalization_handler(self.number_in_str) # corresponding number in word format

    # function that convert the integer part of the numeric number into words format
    # input: numeric number (integer part) (int), output: integer-part number in word format (string)
    def int_to_words(self, number):
        num_to_words = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', \
                        6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', \
                        11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen', \
                        15: 'fifteen', 16: 'sixteen', 17: 'seventeen', 18: 'eighteen', \
                        19: 'nineteen', 20: 'twenty', 30: 'thirty', 40: 'forty', \
                        50: 'fifty', 60: 'sixty', 70: 'seventy', 80: 'eighty', \
                        90: 'ninety', 0: 'zero'}
        try:  # digit in num_to_words dictionary
            return num_to_words[number]
        except KeyError:
            try:  # 2nd digit that not in num_to_words dictionary
                return num_to_words[number - number % 10] + " " + self.int_to_words(number % 10)
            except KeyError:
                try:  # 3rd digit
                    if number % 100 == 0:
                        return num_to_words[(number - number % 100) / 100] + " hundred"
                    else:
                        return num_to_words[(number - number % 100) / 100] + " hundred and " + self.int_to_words(number % 100)
                except:
                    raise ValueError("number is too large") # number > 999

    # function that converts the decimal part of the numeric number into words format
    # input: numeric decimal number (numbers after "point") (string), output: decimal-part number in word format (string)
    def decimal_to_words(self, decimal):  # input should be an str
        decimal_dict = {"1": 'one', "2": 'two', "3": 'three', "4": 'four', "5": 'five', \
                        "6": 'six', "7": 'seven', "8": 'eight', "9": 'nine', "0": "zero"}
        if len(decimal) == 1:
            return decimal_dict[decimal]
        else:
            return self.decimal_to_words(decimal[0]) + " " + self.decimal_to_words(decimal[1:]) # remove the first digit

    # function that separate the numeric number into integer part and decimal part, as they have different word format
    # input: numeric number (string), output: corresponding number in words format (string)
    def number_normalization_handler(self, number_str):
        if "." in number_str:
            return self.int_to_words(int(number_str.split(".")[0])) + " point " + self.decimal_to_words(number_str.split(".")[1])
        else:
            return self.int_to_words(int(number_str))


"""
Date normalization class
object in the class is used to convert dates in format: (D)D/(M)M(/(YY)YY) (string)
into its corresponding word form (string)
working process: convert the date string into and standard date object (handle out of range problems),
then convert dates, month, years to words separately (they have different to-word rules)
and concatenate them into a single string
"""
class DateNormalizer():
    def __init__(self, date_token_str):
        self.date_in_str = date_token_str   # input string in (D)D/(M)M(/(YY)YY) format
        self.normalized_word_tokens = self.date_normalization_handler(self.date_in_str) # corresponding date in word format

    # function that convert a input string in date format into an datetime object
    # purpose: some numbers in format (D)D/(M)M(/(YY)YY) may be invalid, e.g. 32/11/2016 is invalid
    # it will captured when creating datetime object
    # input: date string in format (D)D/(M)M(/(YY)YY) (string), output: datetime object of the given date (datetime object)
    def date_str_to_object(self, date_str):  # normalize date string into one format, raise relevent error if the format is invalid
        if len(date_str.split("/")) == 2:  # only date and month
            return datetime.strptime(date_str, "%d/%m")
        else:  # date, month and year
            if len(date_str.split("/")[2]) == 2:  # year without specifying the century
                return datetime.strptime(date_str, "%d/%m/%y")
            else:  # year with the century specified
                return datetime.strptime(date_str, "%d/%m/%Y")

    # function that convert a numerical date into its corresponding date string in word format
    # input: numerical date(1-31) (int), output: date in words format (string)
    def date_to_words(self, date_number):  # date, not month or year
        date = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', \
                6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth', \
                11: 'eleventh', 12: 'twelfth', 13: 'thirteenth', 14: 'fourteenth', \
                15: 'fifteenth', 16: 'sixteenth', 17: 'seventeenth', 18: 'eighteenth', \
                19: 'nineteenth', 20: ['twentieth', 'twinty'], 30: ['thirtieth', 'thirty']}
        try:  # 1-20
            return date[date_number]
        except KeyError:  # 21-31
            return date[date_number - date_number % 10][1] + " " + self.date_to_words(date_number % 10)

    # function that convert a numerical month into its corresponding month string in word format
    # input: numerical month(1-12) (int), output: month in words format (string)
    def month_to_words(self, month_number):
        month = {1: 'january', 2: 'february', 3: 'march', 4: 'april', 5: 'may', 6: 'june', \
                 7: 'july', 8: 'august', 9: 'september', 10: 'october', 11: 'november', 12: 'december'}
        try:
            return month[month_number]
        except:
            raise ValueError("month is not in correct range")

    # function that convert a numerical year string into its corresponding year string in word format
    # input: numerical year(only consider year 1000 - 9999) (int), output: date in words format (string)
    def year_to_words(self, year_number):  # four digit year in int, only consider year 1000 - 9999
        year_str = str(year_number)
        if not year_str[2] == "0": # xxxx, where x is a digit not equal to 0
            return Number_Normalizer(year_str[0] + year_str[1]).normalized_word_tokens + " " + Number_Normalizer(year_str[2] + year_str[3]).normalized_word_tokens
        else:  # xx0x or x00x or xx00 or x000
            if not year_str[3] == "0": # xx0x or x00x
                if not year_str[1] == "0": # xx0x
                    return Number_Normalizer(year_str[0] + year_str[1]).normalized_word_tokens + " hundred and " + Number_Normalizer(year_str[3]).normalized_word_tokens
                else: # x00x
                    return Number_Normalizer(year_str[0]).normalized_word_tokens + " thousand and " + Number_Normalizer(year_str[3]).normalized_word_tokens
            else:  # xx00 or x000
                if not year_str[1] == "0": # xx00
                    return Number_Normalizer(year_str[0] + year_str[1]).normalized_word_tokens + " hundred"
                else:  # x000:
                    return Number_Normalizer(year_str[0]).normalized_word_tokens + " thousand"

    # function that separate date(numeric), month(numeric), year(numeric), of string in format (D)D/(M)M(/(YY)YY)
    # and produce corresponding date, month, year in words by calling date_to_words, month_to_words, and year_to_words
    # input: numerical date in format (D)D/(M)M(/(YY)YY) (string), output: date string in words (string)
    def date_normalization_handler(self, date_str):
        date_object = self.date_str_to_object(date_str)
        date_in_words = self.date_to_words(date_object.day)
        month_in_words = self.month_to_words(date_object.month)
        if len(date_str.split("/")) == 2:  # only date and month
            return "the {} of {}".format(date_in_words, month_in_words)
        else:  # date, month, year
            year_in_words = self.year_to_words(date_object.year)
            return "the {} of {} {}".format(date_in_words, month_in_words, year_in_words)



if __name__ == "__main__":
    syn_rate = 16000
    S = Synth(wav_folder=args.monophones, rate=syn_rate)

    out = SA.Audio(rate=syn_rate)
    # print out.data, type(out.data) # for testing

    phone_seq = get_phone_seq(args.phrase[0])
    out.data = S.concatenate(phone_seq)

    # data modification
    if args.volume is not None: # ValueError will be handled by SA
        out.rescale(args.volume)
        print "synthesised audio is rescaled by a factor of %.4f" %args.volume

    # output of the modified audio
    if args.play:
        out.play()
    if args.outfile is not None:
        out.save(args.outfile)
        print "synthesised audio is saved at: %s" %args.outfile