"""
This file contains a number of code snippets that were used at some point of the project.
There is no user interface. All the code is separated into functions and that used
whenever needed, mostly to analyse or generate various CSV, JSON, ARFF files.

The following python packages must be installed in order be able to run every function:
- nltk
- sklearn
- scikit-learn
- matplotlib
This can be done with Python package manager (pip).

Author: Mindaugas Ribakauskas, 2093693r
"""

import glob
import os
import codecs
import nltk
import re
import io
import json
import csv
import shutil
import matplotlib.pyplot as plt

from random import randint, shuffle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn import feature_extraction
from collections import OrderedDict
from sklearn.cluster import KMeans
from nltk.tokenize import RegexpTokenizer
from curses.ascii import isdigit
from nltk.corpus import cmudict
from nltk.stem.snowball import SnowballStemmer
from matplotlib import style
style.use("ggplot")



# Tokenizes the text and performs the stemming of each token.
def tokenize_and_stem(text):
    stemmer = SnowballStemmer("english")
    tokens = word_tokenize(text)
    filtered = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered.append(token)
    stems = [stemmer.stem(t) for t in filtered]
    return stems


# Returns the number of syllables in a given word
d = cmudict.dict()
def syll_count(word):
    if word.lower() not in d: # check if a given word is valid
        return -1
    else: # calculate the number of syllables
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]


# Returns two readability metrics for a given text [fig_index AND readability_ease]
def metrics(text):
    sents = nltk.sent_tokenize(text)
    words = RegexpTokenizer(r'\w+').tokenize(text)

    syll = 0 # number of syllables
    diff = [] # difficult words
    for word in words:
        if not re.search(r'\d', word):
            if syll_count(word) >= 3:
                diff += [word]
            if syll_count(word) != -1: # check if the word was valid
                syll += syll_count(word)

    fogIndex = 0.4 * (len(words)/len(sents) + (100 * len(diff)/len(words)))
    readEase = 206.835 - 1.015 * len(words)/len(sents) - 84.6 * syll/len(words)

    return (fogIndex, readEase)


# Returns the statistical information for a given data from a single website.
def corpus_stats(data):
    total_docs = len(data)
    total_words = 0.0
    total_sentences = 0.0
    chars = 0.0
    for doc in data:
        sentences = nltk.sent_tokenize(doc)
        for sent_str in sentences:
            sentence = nltk.word_tokenize(sent_str)
            for token in sentence:
                if re.search('[a-zA-Z]', token): # check if token is a words
                    total_words += 1
                    chars += len(token)
            total_sentences += 1

    # returns  num of docs,  avg. words per doc.,  avg. word length,  avg. sentence length
    return total_docs, total_words/total_docs, chars/total_words, chars/total_sentences


with open('./final_corpus/corpus.json', 'r') as final_c:
    log = json.load(final_c)
    final_c.close()

for l in log:
    d = []
    for f in log[l]['files']:
        loc = './final_corpus/' + str(f['location'])
        with io.open(loc, 'r', encoding="utf-8") as fp:
            d.append(fp.read())
            fp.close()
    num, avg_words, avg_w_len, avg_s_len = corpus_stats(d)
    print l, '&', num, '&', avg_words, '&', avg_w_len, '&', avg_s_len



# Returns the list of file contents for a given log and website name
def read_data(log, name):
    file_data = []
    files = log[name]['files']
    for f in range(len(files)):
        location = str(files[f]['location'])
        with io.open(location, "r", encoding="utf-8") as fp:
            file_data.append(fp.read())
            fp.close()
    return file_data


# Prints the stats of the corpus
def print_stats(log):
    for key in log.keys():
        website = log[key]
        file_data = read_data(log, website['name'])

        # Get the statistics
        docs, tokens, avg_words, avg_sentence = corpus_stats(file_data)
        print website['name'], docs, tokens, avg_words, avg_sentence


# Calculates and prints the average metric scores
def print_metrics(log):
    for key in log.keys():
        website = log[key]
        metric_data = []
        data = read_data(log, website['name'])
        for d in data:
            metric_data += [metrics(d)]

        avg_fog = reduce(lambda x, y: x + y, [i[0] for i in metric_data]) / len(metric_data)
        avg_ease = reduce(lambda x, y: x + y, [i[1] for i in metric_data]) / len(metric_data)
        mean_fog = sorted(metric_data, key=lambda x: x[0])[len(metric_data)/2][0]
        mean_ease = sorted(metric_data, key=lambda x: x[1])[len(metric_data)/2][1]

        print "%s & %.2f & %.2f & %.2f & %.2f" % (website['name'], avg_fog, mean_fog, avg_ease, mean_ease)


# Returns 3 lists of lacations to the files divided by document metric scores
def divide_by_metrics(website_names):
    easy = []
    medium = []
    hard = []
    for name in website_names:
        website = log[name]
        data = read_data(log, name)
        files = [ f['location'] for f in website['files']]
        metric_data = []
        for i in range(len(files)):
            metric_data += [(files[i], metrics(data[i]))]
        sorted_list = sorted(metric_data, key=lambda x: x[1][0])
        sorted_list = [x[0] for x in sorted_list]
        em_cutoff = int(len(sorted_list)/3.0)
        mh_cutoff = int(len(sorted_list)/3.0*2.0)
        easy += sorted_list[:em_cutoff]
        medium += sorted_list[em_cutoff:mh_cutoff]
        hard += sorted_list[mh_cutoff:]
    return easy, medium, hard


# Sceletone structure for the clustered corpus.
clustered = {
    'easy': {
        'websites': ['pitara', 'timeForKids', 'teachingKidsNews', 'tweenTribune'],
        'd1': { 'c1': [], 'c2': [], 'c3': [] },
        'd2': { 'c1': [], 'c2': [], 'c3': [] },
        'd3': { 'c1': [], 'c2': [], 'c3': [] }
    },
    'medium': {
        'websites': ['nme', 'newScientist', 'dogo'],
        'd1': { 'c1': [], 'c2': [], 'c3': [] },
        'd2': { 'c1': [], 'c2': [], 'c3': [] },
        'd3': { 'c1': [], 'c2': [], 'c3': [] }
    },
    'hard': {
        'websites': ['guardian', 'reuters', 'theConversation'],
        'd1': { 'c1': [], 'c2': [], 'c3': [] },
        'd2': { 'c1': [], 'c2': [], 'c3': [] },
        'd3': { 'c1': [], 'c2': [], 'c3': [] }
    }
}


# Divides the documents from the given set of locations into 3 clusters.
def divide_by_clusters(locations):
    data = []
    for location in locations:
        with io.open(str(location), "r", encoding="utf-8") as fp:
            data.append(fp.read())
            fp.close()

    # Vectorize
    vec = TfidfVectorizer(max_df=0.8, max_features=200000,
                          min_df=0.2, stop_words='english',
                          use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
    tfidf_matrix = vec.fit_transform(data)

    # Cluster
    kmeans = KMeans(n_clusters=3)
    kmeans.fit_transform(tfidf_matrix)
    labels = kmeans.labels_
    clusters = {}

    # Separate by clusters
    for i in range(len(labels)):
        c = labels[i]
        if c not in clusters:
            clusters[c] = []
            clusters[c] += [locations[i]]
        else:
            clusters[c] += [locations[i]]

    return clusters[0], clusters[1], clusters[2]


# Divides the entire corpus into 9 categories and clusters each category into 3 clusters.
def divide_and_cluster(clustered):
    path = './clustered/'
    if not os.path.exists(path):
        os.makedirs(path)

    for group_name in clustered.keys():
        # Create a directory
        if os.path.exists(path + group_name):
            shutil.rmtree(path + group_name)
        os.makedirs(path + group_name)

        group = clustered[group_name]
        d1, d2, d3 = divide_by_metrics(group['websites'])

        # For difficulty level in 3 levels within the group
        for d in (('d1', d1), ('d2', d2), ('d3', d3)):
            p = path + group_name + '/' + d[0] + '/'
            d_group = group[d[0]]
            if os.path.exists(p): # remove the old files, if there are any
                shutil.rmtree(p)
            os.makedirs(p)

            # Cluster documents within the group into 3 clusters
            c1, c2, c3 = divide_by_clusters(d[1])
            for c in (('c1', c1), ('c2', c2), ('c3', c3)):
                d_group[c[0]] = c[1]
                if os.path.exists(p + c[0]): # remove the old files, if there are any
                    shutil.rmtree(p + c[0])
                os.makedirs(p + c[0])

                # Copy all the documents from the cluster into a certain folder
                for f_location in c[1]:
                    dest = f_location.split('/')[3]
                    dest = p + c[0] + '/' + dest
                    shutil.copyfile(f_location, dest)

    # Write metadata into a file
    with open('clustered.json', 'w') as cluster_log:
        json.dump(clustered, cluster_log)
        cluster_log.close()


# Generates a single test case for the crowd-sourcing.
taken = [] # list of files that were already used
def generate_test_case(t1, t2, t3):
    with open('clustered.json', 'r') as clustered_log_file:
        clustered_log = json.load(clustered_log_file)
        clustered_log_file.close()

    data = {}
    for t in (t1, t2, t3):
        t = t.split('_')
        group = clustered_log[t[0]][t[1]]
        for c in ('c1', 'c2', 'c3'):
            files = group[c]
            i = 0
            while i < 3:
                f_name = files[randint(0, len(files)-1)]
                if f_name not in taken: # check for duplicates
                    taken.append(f_name)

                    t_id = '_'.join(str(x) for x in [f_name.split('/')[3].split('.')[0], t[0], t[1], c])

                    with open(f_name, 'r') as f:
                        text = f.read()
                        f.close()

                    # Tokenize the text and count the word quantities
                    tokens = RegexpTokenizer('\w+').tokenize(text)
                    words = []
                    for word in tokens:
                        word = word.lower()
                        if re.search('^[A-Za-z]*$', word):
                            if word not in stopwords.words('english') and len(word) > 3:
                                words.append(word)

                    word_count = {}
                    for word in words:
                        if word in word_count:
                            word_count[word] += 1
                        else:
                            word_count[word] = 1
                    word_list = []
                    for word in word_count.keys():
                        count = word_count[word]
                        word_list += [(word, count)]

                    words_sorted = sorted(word_list, key=lambda x: x[1])
                    if t[0] in data:
                        data[t[0]] += [(t_id, text.replace('"', '""'), words_sorted[-15:])]
                    else:
                        data[t[0]] = []
                        data[t[0]] += [(t_id, text.replace('"', '""'), words_sorted[-15:])]
                    i += 1


    # Shuffle the list to make the order more random
    for k in data.keys():
        shuffle(data[k])

    string = []
    for i in range(0, 9):
        line = []
        line.append('_'.join([t1[0]+t1[-1], t2[0]+t2[-1], t3[0]+t3[-1]]))
        keys = data.keys()
        shuffle(keys)
        for t in keys:
            line.append(data[t][i][0])
            line.append('"' + data[t][i][1] + '"')
            for w in data[t][i][2]:
                line.append(w[0])
                line.append(w[1])
        l = ','.join(str(x) for x in line)
        string.append(l + '\n')

    f_string = ''.join(string)
    return f_string


# Generaet a complete CSV test file to be used during the crowd-sourcing
def generate_csv():
    strings = []

    # Add the header
    header = []
    header.append('type')
    header.append('t1_id')
    header.append('t1')
    for i in range(1, 16):
        header.append('t1_w' + str(i))
        header.append('t1_w' + str(i) + '_val')
    header.append('t2_id')
    header.append('t2')
    for i in range(1, 16):
        header.append('t2_w' + str(i))
        header.append('t2_w' + str(i) + '_val')
    header.append('t3_id')
    header.append('t3')
    for i in range(1, 16):
        header.append('t3_w' + str(i))
        header.append('t3_w' + str(i) + '_val')

    strings.append(','.join(header) + '\n')

    # Add all the remaining rows

    # Corner cases
    strings.append(generate_test_case('easy_d1', 'medium_d1', 'hard_d1'))
    strings.append(generate_test_case('easy_d3', 'medium_d3', 'hard_d3'))

    # Group cases
    strings.append(generate_test_case('easy_d1', 'easy_d2', 'easy_d3'))
    strings.append(generate_test_case('medium_d1', 'medium_d2', 'medium_d3'))
    strings.append(generate_test_case('hard_d1', 'hard_d2', 'hard_d3'))

    # Middle cases
    strings.append(generate_test_case('easy_d2', 'medium_d2', 'hard_d2'))
    strings.append(generate_test_case('easy_d2', 'medium_d1', 'hard_d1'))
    strings.append(generate_test_case('easy_d3', 'medium_d2', 'hard_d1'))
    strings.append(generate_test_case('easy_d3', 'medium_d2', 'hard_d2'))
    strings.append(generate_test_case('easy_d3', 'medium_d3', 'hard_d2'))

    # Write to file
    final_string = ''.join(strings)
    with open('data.csv', 'w') as fp:
        fp.write(final_string)
        fp.close()


# Calculates the averages of the results from crowd-sourcing and generates a plot.
def plot_results():

    stats = {}

    # Read the data from csv file.
    with open('agr.csv', 'r') as csv_file:
        res = csv.DictReader(csv_file)

        for row in res:
            id1 = '_'.join(row['t1_id'].split('_')[2:4])
            id1 = id1[0]+id1[-1]
            id2 = '_'.join(row['t2_id'].split('_')[2:4])
            id2 = id2[0]+id2[-1]
            id3 = '_'.join(row['t3_id'].split('_')[2:4])
            id3 = id3[0]+id3[-1]

            l1 = row['grade_level_text_1'].split('_')[1]
            l2 = row['grade_level_text_2'].split('_')[1]
            l3 = row['grade_level_text_3'].split('_')[1]

            r1 = row['text_1']
            r2 = row['text_2']
            r3 = row['text_3']

            for (i, r, l) in [(id1, float(r1), float(l1)), (id2, float(r2), float(l2)), (id3, float(r3), float(l3))]:
                if i not in stats:
                    stats[i] = [0,0,0]
                else:
                    stats[i][0] += r
                    stats[i][1] += l
                    stats[i][2] += 1

    # Calculate the averages
    for i in stats.keys():
        stats[i][0] = stats[i][0]/float(stats[i][2])
        stats[i][1] = stats[i][1]/float(stats[i][2])

    colors = {
        'e1': '#00cc00', 'e2': '#00cc00', 'e3': '#00cc00',
        'm1': '#0066ff', 'm2': '#0066ff', 'm3': '#0066ff',
        'h1': '#ff0000', 'h2': '#ff0000', 'h3': '#ff0000'}

    # Plot the graph.
    for i in stats:
        color = colors[i]
        if i[0] == 'e':
            label = 'Easy'
        if i[0] == 'm':
            label = 'Medium'
        if i[0] == 'h':
            label = 'Hard'
        plt.plot(stats[i][1], stats[i][0],
                marker='o',
                label=label,
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=10)
        plt.text(stats[i][1]+0.01, stats[i][0]+0.01, i.upper(), size=20)

    ax = plt.gca()
    # ax.set_title("Plot Name")
    ax.set_xlabel('Average Difficulty Level')
    ax.set_ylabel('Average Rank')

    plt.rcParams['legend.handlelength'] = 0 # removes the line from the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), numpoints=1)
    plt.show()



# Adds the categories to the log files
def addCategories():

    g1 = []
    g2 = []
    g3 = []
    g4 = []

    with open('./clustered.json', 'r') as cl_file:
        clustered = json.load(cl_file)
        cl_file.close()

    g1 += clustered['easy']['d1']['c1'] + clustered['easy']['d1']['c2'] + clustered['easy']['d1']['c3']
    g1 += clustered['easy']['d2']['c1'] + clustered['easy']['d2']['c2'] + clustered['easy']['d2']['c3']

    g2 += clustered['medium']['d1']['c1'] + clustered['medium']['d1']['c2'] + clustered['medium']['d1']['c3']
    g2 += clustered['easy']['d3']['c1'] + clustered['easy']['d3']['c2'] + clustered['easy']['d3']['c3']
    g2 += clustered['medium']['d2']['c1'] + clustered['medium']['d2']['c2'] + clustered['medium']['d2']['c3']

    g3 += clustered['hard']['d1']['c1'] + clustered['hard']['d1']['c2'] + clustered['hard']['d1']['c3']
    g3 += clustered['medium']['d3']['c1'] + clustered['medium']['d3']['c2'] + clustered['medium']['d3']['c3']

    g4 += clustered['hard']['d2']['c1'] + clustered['hard']['d2']['c2'] + clustered['hard']['d2']['c3']
    g4 += clustered['hard']['d3']['c1'] + clustered['hard']['d3']['c2'] + clustered['hard']['d3']['c3']

    logs = glob.glob('./data/logs/*.json')
    for log_name in logs:
        with open(log_name, 'r') as lf:
            log = json.load(lf)
            lf.close()
        files = log['files']
        for f in files:
            if f['location'] in g1:
                f['category'] = 'level_1'
            elif f['location'] in g2:
                f['category'] = 'level_2'
            elif f['location'] in g3:
                f['category'] = 'level_3'
            elif f['location'] in g4:
                f['category'] = 'level_4'

        with open(log_name, 'w') as lf:
            json.dump(log, lf, indent=2)


# Generates ARFF file to be used in Weka.
def createArffFile():

    features = {}

    # Read the names of all the features
    with open('./data/logs/pitara_log.json') as fp:
        l = json.load(fp)
        fp.close()

        ft = l['files'][0]['features']
        for cl in ft.keys():
            if cl not in features:
                features[cl] = []
            for f in ft[cl].keys():
                features[cl].append(f)

    #File beginning
    arff = "@relation difficulty\n"
    for cl in features['syntacticFeatures']:
        arff += '@attribute ' + cl + ' numeric\n'
    for cl in features['lexicalFeatures']:
        arff += '@attribute ' + cl + ' numeric\n'
    for cl in features['posFeatures']:
        arff += '@attribute ' + cl + ' numeric\n'
    for cl in features['discourseFeatures']:
        arff += '@attribute ' + cl + ' numeric\n'
    arff += '@attribute class {level_1,level_2,level_3,level_4}\n\n@data\n'


    lines = []

    logs = glob.glob('./data/logs/*.json')

    for log_name in logs:
        with open(log_name, 'r') as lf:
            log = json.load(lf)
            lf.close()
        files = log['files']
        for f in files:
            line = []

            synF = f['features']['syntacticFeatures']
            for ft in features['syntacticFeatures']:
                line.append(str(synF[ft]))

            lexF = f['features']['lexicalFeatures']
            for ft in features['lexicalFeatures']:
                line.append(str(lexF[ft]))

            posF = f['features']['posFeatures']
            for ft in features['posFeatures']:
                line.append(str(posF[ft]))

            disF = f['features']['discourseFeatures']
            for ft in features['discourseFeatures']:
                line.append(str(disF[ft]))

            line.append(str(f['category']))
            line = ','.join(line)
            lines.append(line)

    shuffle(lines)

    for line in lines:
        arff += line + '\n'

    with open('test.arff', 'w') as af:
        af.write(arff)
        af.close


# Generates the final corpus corpus with 4 difficulty levels.
def generate_final_corpus():

    logs = glob.glob('./data/logs/*.json')

    final = {}

    path = './corpus/'
    if not os.path.exists(path):
        os.makedirs(path)

    for log_name in logs:

        with open(log_name, 'r') as log_file:
            log = json.load(log_file)
            log_file.close()
        files = log['files']
        for f in files:
            if not os.path.exists(path + f['category']):
                os.makedirs(path + f['category'])

            if f['category'] not in final:
                final[f['category']] = {
                    'name': f['category'],
                    'numberOfDocuments': 0,
                    'files': []
                }
            else:
                new_file = {
                    'name': f['name'],
                    'location': './corpus/' + f['category'] + '/' + f['name'] + '.txt',
                    'topic': f['topic'],
                    'features': f['features']
                }
                final[f['category']]['numberOfDocuments'] += 1
                final[f['category']]['files'].append(new_file)
                shutil.copyfile(f['location'], new_file['location'])

    with open('./corpus.json', 'w') as final_json:
        json.dump(final, final_json, indent=2)
        final_json.close()
