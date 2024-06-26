# Data-Extraction-and-NLP
A brief walkthrough about web scraping and text analysis.

## Output Data Explanation

### Positive Score
This score is calculated by assigning the value of +1 for each word found in the Positive Dictionary and then adding up all the values.

### Negative Score
This score is calculated by assigning the value of -1 for each word found in the Negative Dictionary and then adding up all the values. We multiply the score by -1 so that the score is a positive number.

### Polarity Score
This score determines if a given text is positive or negative in nature. It is calculated using the formula:

Polarity Score = (Positive Score – Negative Score)/ ((Positive Score + Negative Score) + 0.000001)

The range is from -1 to +1.

### Subjectivity Score
This score determines if a given text is objective or subjective. It is calculated using the formula:

Subjectivity Score = (Positive Score + Negative Score)/ ((Total Words after cleaning) + 0.000001)

The range is from 0 to +1.

## Analysis of Readability

### Gunning Fog Index
The analysis of readability is calculated using the Gunning Fog index formula described below:

Average Sentence Length = the number of words / the number of sentences
Percentage of Complex words = the number of complex words / the number of words 
Fog Index = 0.4 * (Average Sentence Length + Percentage of Complex words)


### Average Number of Words Per Sentence
The formula for calculating this is:

Average Number of Words Per Sentence = the total number of words / the total number of sentences


### Complex Word Count
Complex words are words in the text that contain more than two syllables.

### Word Count
We count the total cleaned words present in the text by:
- Removing the stop words (using the `stopwords` class of the `nltk` package).
- Removing any punctuations like ?, !, , . from the word before counting.

### Syllable Count Per Word
We count the number of syllables in each word of the text by counting the vowels present in each word. We also handle some exceptions like words ending with "es" or "ed" by not counting them as a syllable.

### Personal Pronouns
To calculate personal pronouns mentioned in the text, we use regex to find the counts of the words: “I,” “we,” “my,” “ours,” and “us”. Special care is taken so that the country name US is not included in the list.

### Average Word Length
Average word length is calculated by the formula:

Sum of the total number of characters in each word/Total number of words
