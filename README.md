# Blackcoffer's-Web-Articles-Analysis-Using-Text-Mining-and-NLP
## **Objective**
The goal of this project is to retrieve textual articles from the BlackCoffer website and apply Natural Language Processing (NLP) techniques to analyze the text. The objective includes computing various variables listed below through text analysis.
The methodlogy is divided into 2 parts:
1. Web-Scraping
2. Natural Language Processing
## **Variables**
1. Number_of_Sentences
2. Number_of_Words
3. Average_Sentence_Length: The number of words / the number of sentences
4. Syllable_Count
5. Complex_Count: Complex words are words in the text that contain more than two syllables.
6. Percentage_of_Complex_words
7. Fog_Index: 0.4 * (Average Sentence Length + Percentage of Complex words)
8. Positive_Score: This score is calculated by assigning the value of +1 for each word if found in the Positive Dictionary and then adding up all the values.
9. Negative_Score: This score is calculated by assigning the value of -1 for each word if found in the Negative Dictionary and then adding up all the values. We multiply the
   score with -1 so that the score is a positive number.
10. Polarity_Score: (Positive Score – Negative Score)/ ((Positive Score + Negative Score) + 0.000001)
11. Subjective_Score: (Positive Score + Negative Score)/ ((Total Words after cleaning) + 0.000001)

## **Software and Libraries**
<img src="https://github.com/KolanHarsha/DDos-detection-Using-Machine-Learning/assets/110462466/ec05c02a-389a-4363-8b8c-9b1ba8ca28b0" alt="python" width="150" height="100">
<img src="https://github.com/KolanHarsha/DDos-detection-Using-Machine-Learning/assets/110462466/88e29b73-06a2-48ac-8e80-0cd755dd980e" alt="jup" width="150" height="100">
<img src="https://github.com/KolanHarsha/Blackcoffers-Web-Articles-Analysis-Using-Text-Mining-and-NLP/assets/110462466/8a00d332-c459-4d8f-b002-b3d5d675daa7" alt="python" width="150" height="100">
<img src="https://github.com/KolanHarsha/Blackcoffers-Web-Articles-Analysis-Using-Text-Mining-and-NLP/assets/110462466/9e5fe252-cf00-4466-b05e-d184f64404bf" alt="python" width="150" height="100">

## **How to run**
The spreadsheet input_links.xlsx contains a total of 89 links, which we will utilize to scrape the websitesand extract the Blackcoffer text articles using the BeautifulSoup library.

## **Part 1- Web-Scraping**
The Python script for web scraping can be found in the "Web_Scraping.ipynb" file. Upon execution of this file, it will generate a text file named "Blackcoffer_insights.txt" containing all the textual content from the 89 web articles.
## **Part 2-Natural Language Processing**
The Python notebook "main.ipynb" encompasses all the necessary natural language processing techniques essential for analyzing textual articles.

### **Stop-Words**
The words that are removed during the data preprocessing step because they carry minimal meaning in a sentence are stored in the StopWords directory.
```bash
import os
folder_path = r"C:\Users\STSC\Desktop\Data Analytics Project\StopWords"
stop_words=[]
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):  
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            file_content = file.readlines()
        file_data=[line.split('|')[0].strip() for line in file_content]
        stop_words.extend(file_data)
```
The code snippet above is used to read all the stopwords text files in the 'StopWords' directory and store them in a list.

### **Positive-Words**
All the positive words used for this project are available in positive-words.txt file
```bash
file_path=r"C:\Users\STSC\Desktop\Data Analytics Project\positive-words.txt"
with open(file_path, 'r') as file:
        file_content = file.readlines()
data=[line.split('|')[0].strip() for line in file_content]
positivewords=[]
for i in data:
    if i not in stopwords:
        positivewords.append(i)
```
The code snippet above is used to read all the positive-words in the 'positive-words.txt' file and store them in a list.

### **Negative-Words**
All the Negative words used for this project are available in negative-words.txt file
```bash
file_path=r"C:\Users\STSC\Desktop\Data Analytics Project\negative-words.txt"
with open(file_path, 'r') as file:
        file_content = file.readlines()
data=[line.split('|')[0].strip() for line in file_content]
negativewords=[]
for i in data:
    if i not in stopwords:
        negativewords.append(i)
```
The code snippet above is used to read all the negative-words in the 'negative-words.txt' file and store them in a list.

### **Tokenization**
The text data generated after web-scraping is stored in a dictionary with Page-titles as keys and List of paragraphs as it's values.
```bash
for key, value in BC.items():          # BC={A:[a,b,c]}
    words = []
    for i in value:
        w = [[x for x in word_tokenize(sent)] for sent in sent_tokenize(i)]
        wordlist = sum(w, [])
        words.extend(wordlist)
```
The code snippet above is used to tokenize the paragraphs and store them in a list.
### **Functions to compute the variables**
Positive-Score:
```bash
def positive_score(x):
    count=0
    for i in x:
        if i in positivewords:
            count=count+1
    return count
```
Negative-Score:
```bash
def negative_score(x):
    count=0
    for i in x:
        if i in negativewords:
            count=count+1
    return count
```
Polarity-Score:
```bash
def polarity_score(x,y):
    z = (x - y)/ ((x + y) + 0.000001)
    return z
```
Subjective-Score:
```bash
def subjective_score(x,y,l):
    z=(x + y)/ ((l) + 0.000001)
    return z
```
Counting Syllables:
```bash
def count_syllables(x):
    vowel_pattern = re.compile(r'[aeiouAEIOU]+')
    y = []
    for i in x:
        # Handling exceptions for words ending with "es" or "ed"
        if i.endswith(("es", "ed")):
            syllable_count = len(re.findall(vowel_pattern, i[:-2]))
        else:
            syllable_count = len(re.findall(vowel_pattern, i))
        y.append(syllable_count)
    syllable_counts=0
    for j in range(0,len(y)):
        syllable_counts=y[j]+syllable_counts
        
    return syllable_counts
```
Counting Syllables:
```bash
def count_syllables(x):
    vowel_pattern = re.compile(r'[aeiouAEIOU]+')
    y = []
    for i in x:
        # Handling exceptions for words ending with "es" or "ed"
        if i.endswith(("es", "ed")):
            syllable_count = len(re.findall(vowel_pattern, i[:-2]))
        else:
            syllable_count = len(re.findall(vowel_pattern, i))
        y.append(syllable_count)
    syllable_counts=0
    for j in range(0,len(y)):
        syllable_counts=y[j]+syllable_counts
        
    return syllable_counts
```
Counting Complex-words
```bash
import re
def complex_count(x):
    vowel_pattern = re.compile(r'[aeiouAEIOU]+')
    y = []
    for i in x:
        # Handling exceptions for words ending with "es" or "ed"
        if i.endswith(("es", "ed")):
            syllable_count = len(re.findall(vowel_pattern, i[:-2]))
        else:
            syllable_count = len(re.findall(vowel_pattern, i))
        y.append(syllable_count)
    complex_counts = 0
    for j in range(0, len(y)):
        if y[j] > 2:
            complex_counts += y[j]  # Accumulate syllable counts greater than 2
        
    return complex_counts
```
Running the main.ipynb file till here will generate a 'output.xlsx' file which contains the computed variables for all the 89 links.

## **Analyzing-Results**
### **Chunking**
Suppose we want to know what the top-positive article and top-negative article speak about? Instead of reading the entire article we can use chunking to get the chunks of the paragraphs and retrieve information from those chunks.
Chunkstyle:
```bash
'''Chunk1:{<VB.?>+<DT|JJ.?|PRP.?>*<NN.*>+}
    Chunk2:{<Chunk1><DT|JJ.?|PRP.?|JJ.?>*<NN.*>+} 
    '''
```
1. After employing this chunking style of breaking down information into smaller chunks, I found that the article with the highest positive score delves into the realm of online marketing. It discusses how major companies such as Facebook utilize online marketing strategies to promote their products and effectively reach their desired audience.
2. Likewise, the negative article addresses the effects of the Covid-19 pandemic on the environment.

## **Contributors**
- Sai Harsha Vardhan Reddy, Kolan- skolan@horizon.csueastbay.edu, harsha62334@gmail.com

Thanks for reading!





