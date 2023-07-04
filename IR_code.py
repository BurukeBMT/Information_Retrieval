import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
import pdfplumber
#nltk.download('punkt')
#nltk.download('stopwords')


#------------------------vector space model--------------------------------
class InvertedIndex:
    def __init__(self):
        self.documents = {}
        self.index = {}
        self.vectorizer = TfidfVectorizer()

    def add_document(self, doc_id, text):
        self.documents[doc_id] = text
        tokens = self._process_text(text)
        term_positions = {}
        
        for i, token in enumerate(tokens):
            if token not in self.index:
                self.index[token] = {
                    'DF': 1,
                    'postings': {doc_id: {'TF': 1, 'positions': [i]}}
                }
            else:
                self.index[token]['DF'] += 1
                if doc_id in self.index[token]['postings']:
                    self.index[token]['postings'][doc_id]['TF'] += 1
                    self.index[token]['postings'][doc_id]['positions'].append(i)
                else:
                    self.index[token]['postings'][doc_id] = {'TF': 1, 'positions': [i]}
            
            term_positions[token] = self.index[token]['postings'][doc_id]['positions']
        with open('posting.txt', 'w')as file:
            file.write('\n')
            file.write("{:<20} {:<15} {}".format('Terms','frequency','locations'))
            file.write('\n')
            file.write('\n')
            for key,value in term_positions.items():
                terms = str(key)
                apperance = str(value)
                file.write("{:<20} {:<15} {}".format(terms,str(len(value)),apperance))
                file.write('\n')
        
        return term_positions
    def _process_text(self, text):
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
        return tokens

    def fit_vectorizer(self):
        corpus = list(self.documents.values())
        self.vectorizer.fit(corpus)

    def retrieve_documents(self, query):
        query_vector = self.vectorizer.transform([query])
        similarity_scores = cosine_similarity(query_vector, self.vectorizer.transform(list(self.documents.values())))[0]
        results = [(doc_id, score) for doc_id, score in zip(self.documents.keys(), similarity_scores)]
        results = sorted(results, key=lambda x: x[1], reverse=True)
        return results

    def get_document_text(self, doc_id):
        return self.documents[doc_id]

    def get_term_positions(self, term, doc_id):
        if term in self.index and doc_id in self.index[term]['postings']:
            return self.index[term]['postings'][doc_id]['positions']
        else:
            return []
ir = InvertedIndex()
#--------------------------------file extracter extracts file from folder----------------------------------------------------

def extract_file(folder_path):
    files = []
    file_dictionary = {}
    file_no = 1
    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        file_dictionary[file_no] = file_path
        files.append(file_path)
        file_no += 1
    return (files, file_dictionary)

#------------------------------------determines the file type--------------------------------------
def get_file_type(path):
    _, file_extention = os.path.splitext(path)
    if file_extention == '.txt':
        return 'text'
    elif file_extention == '.pdf':
        return 'pdf'
    else:
        return 'unsupported file'
#-------------------------------------convert file any to text--------------------------
def txter(files):
    type = get_file_type(files)
    if type == 'text':
        with open(files, 'r') as file:
            text = file.read()
        return text
    elif type == 'pdf':
        with pdfplumber.open(files) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()

        return text
    else:
        return '!!!  invalid file type please enter only text | pdf | word files'
        #--------------------------------------tokenizer--------------------------------------
def tokenizer(files):
    type = get_file_type(files)
    if type == 'text':
        with open(files, 'r') as file:
            text = file.read()
        tokens = word_tokenize(text)
        return tokens
    elif type == 'pdf':
        with pdfplumber.open(files) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()

        tokens = word_tokenize(text) + [str(files)]
        return tokens
    else:
        return '!!!  invalid file type please enter only text | pdf | word files'
def tokenize(files):
    token_list = []
    for file in files:

        token_list.append(tokenizer(file))
    return token_list
# ---------------------stemmer----------------------------
def stemm(files):
    # before stemmming first we have to tokenizw the words
    token_list = tokenize(files)
    # the token list is two dimentional list
    stemmer = PorterStemmer()

    stemmed_word_list = []
    for tokens in token_list:
        stemmed_words = []
        for word in tokens:
            stemmed_word = stemmer.stem(word)
            stemmed_words.append(stemmed_word)
        stemmed_word_list.append(stemmed_words)
    return stemmed_word_list
# -------------------------stopword removal---------------------------
def stopword_removal(files):
    sw_list = stemm(files)
    stop_words = set(stopwords.words('english'))
    result = []
    l = ["''", "``", "'s"]
    for lists in sw_list:
        lis = []
        for i in lists:
            if (i.casefold() not in stop_words) and (len(i) > 1) and (i.casefold() not in l):
                lis.append(i)
        result.append(lis)
    # print(result)
    return result
#--------------------------------------------------vocabuary builder-----------------------------------(inverted file)
def vocabulary_file(folder_path):
    files, file_dictionary = extract_file(folder_path)
    final_index_list = stopword_removal(files)
    invertd_term_list = []
    doc_id = 1
    for i in final_index_list:
        for j in i:
            freq = i.count(j)
            invertd_term_list.append(j + " " + str(doc_id) + " " + str(freq))
        doc_id += 1
    result = list(set(invertd_term_list))
    return(sorted(result), file_dictionary)
# ---------------------------------------------locaton finder--------------------------------------------
def find_character_location(file_path, character):
    content = txter(file_path)
    index = content.find(character)  # Using the 'find()' function
    return index
def searcher(file_path, character):
    with open(file_path, 'r') as file:
        content = file.read()

        if character in content.lower():
            return 1
        else:
            return 0  # Using the 'find()' function
# this block writes the vocabulary file --------------------------------------------
x, y = vocabulary_file('docs/')
file = open('inverted.txt', 'w')
file.write(("{:<15} {:<8} {:<8} {:<15}".format(
    'indecx_term', 'DOC#', 'Freq', 'location'))+'\n\n')
for i in x:
    z = i.split()
    location = find_character_location(y[int(z[1])], z[0])
    content = "{:<15} {:<8} {:<8} {:<15}".format(
        i.split()[0], i.split()[1], i.split()[2], location)
    file.write(content+"\n")
#------------------------------------------------------------------retrieve the documents---------------------------------------------------
files, dictionary = extract_file('docs/')
x=1
for i in files:
    ir.add_document(x,txter(i))
    x+=1
ir.fit_vectorizer()
query = input("\n\nEnter your queries | what you are searching for (^_^) ? : ").strip()
retrieved_docs = ir.retrieve_documents(query)
cout = 0
output =[]
for doc_id, sim in retrieved_docs:
    if sim != 00:
        out =[]
        doc_text = ir.get_document_text(doc_id)
        term_positions = ir.get_term_positions(query, doc_id)
        out.append(f"Document ID----------------------------->: {doc_id}, Similarity: {sim:.2f}")
        out.append(f"Follow this link to read the document--->: {dictionary[doc_id]}")
        out.append(f"The terms found at---------------------->: {term_positions}")
        #print()
        cout +=1
        output.append(out)
    else :
        continue
if cout == 0:
    print ("\n\n(^_^) sorry no document matchs your query ! try another query or privede me a document collection.\n\n")

else :
    print("\n\nThe following items matchs you query:")
    rnak = 1
    for i in output:
        print ("\n\nDoc found |Rnak "+str(rnak))
        for j in i:
            print (j)
        rnak+=1