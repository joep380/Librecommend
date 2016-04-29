from __future__ import division
from random import randint
import numpy as np
import random
import sys
import codecs
import os
import re
import math
import time
from flask import Flask, url_for, request, render_template, redirect, make_response
from app import app

def word_count_dict(folder_path, review_list, index):
    word_count = {}
    #check_q = get_query(q)
    pathname = os.path.join(folder_path, review_list[index])
    input_file = open(pathname, 'r')
    doc = input_file.read()
    #d_l = doc.split('\n')
    #print "title: " + d_l[0]
    #print "description: " + d_l[1]
    text = doc #d_l[1]
    input_file.close()
    words = re.split('\W+', text)
    for word in words:
      word = word.lower()
      if len(word) >= 3:
        if word.isalpha():
          #if word in check_q:    
          if not word in word_count:
            word_count[word] = 1
          else:
            word_count[word] = word_count[word] + 1           
    return word_count

def create_tfidf(folder_path):
  wcd_list = []
  #check_q = get_query(q)
  idf_list = []
  review_list=os.listdir(folder_path)
  for index in range(len(review_list)):
    wcd_list.append(word_count_dict(folder_path, review_list, index))
  df_dic = {}
  #for word in check_q: #set df vals to zero 
   # df_dic[word] = 0
  for dict in wcd_list: #get df values
    for key in dict:
      if key not in df_dic:
        df_dic[key] = 1 #first document occurence
      else:
        df_dic[key] = df_dic[key] + 1
  for dict in wcd_list:
    idf_dic = {}
    for key in dict:
      #if key in df_dic:
      idf_dic[key] = df_dic[key]
    idf_list.append(idf_dic)    
  #get tf component
  for dict in wcd_list:
    for key in dict:
      dict[key] = 1 + math.log10(dict[key])
  for dict in idf_list: #get idf component
    for key in dict:
      dict[key] = math.log10(len(review_list) / dict[key])
  #get tfidf values using wcd_list and idf_list
  #sort dicts to get list components aligned
  list_tfidf = []
  for dict in idf_list: #get idf values in 
    tfidf_dict = {}
    for key in sorted(dict):
      tfidf_dict[key] = dict[key]
    list_tfidf.append(tfidf_dict)     
  v = 0
  for dict in wcd_list:    # multiply tf values to the idf values
    for key in sorted(dict):
      list_tfidf[v][key] = list_tfidf[v][key] * dict[key]
    v = v + 1
  #get tfidf values
  return list_tfidf 
 
def find_norm(list_tfidf):
  # norm for doc = sqrt( summation( tfidf_vals^2 ) ) 
  use_tfd = list_tfidf
  sums_list = []
  fd_list = []
  for dict in use_tfd:
    sum_d = 0
    for key in sorted(dict):
      sum_d = sum_d + math.pow(dict[key] , 2)
    sum_d = math.sqrt(sum_d)
    sums_list.append(sum_d)
  i = 0
  for dict in use_tfd:
    for key in sorted(dict):
      if dict[key] != 0:
        dict[key] = dict[key] / sums_list[i]        
    i = i + 1
  return use_tfd  #returns normalized tfidf values

def norm_q(q):
  use_q = get_query(q)
  sum = 0
  for key in use_q:
    sum = sum + math.pow(use_q[key] , 2)
  norm_factor = math.sqrt(sum)
  for key in use_q:
    use_q[key] = use_q[key] / norm_factor 
  return use_q # get normalized query values

def get_count(tup): #helper for sorting
  return tup[1]
  
def find_doc_tf_score(tfidf_dict_list, folder_path):
  scores = tfidf_dict_list
  #for dict in scores:
  #  for key in dict:
  #    for key2 in normed_q_list:
  #      if     key == key2 :
  #        dict[key] = dict[key] * normed_q_list[key2]
  score_list = []
  for dict in scores:
    doc_score = 0
    for key in dict:
      #for key2 in normed_q_list:
      #  if key == key2:
          doc_score = doc_score + dict[key]
    score_list.append(doc_score)
  #relate files to scores
  file_dict = {}
  review_list=os.listdir(folder_path)

  for index in range(len(review_list)):
    #pathname = os.path.join(folder_path, review_list[index])
    #input_file = open(pathname, 'r')
    #doc = input_file.read()
    #d_l = doc.split('\n')
    #print "title: " + d_l[0]
    #print "description: " + d_l[1]
    title = review_list[index]#d_l[0]
    #input_file.close()
    file_dict[title] = score_list[index]
  #sort file scores
  items = sorted(file_dict.items(), key=get_count, reverse=True)
  #print "The top five scoring files (highest to lowest) based on doc tfidf are: "
  #num = 1
  #for item in items[:5]:
  #  print str(num)+". ", "Title: ", item[0], " Score: ", item[1] #print results 
  #  num = num + 1
  #for item in items:
  #    print item[0]
  #    print item[1]
  return items  
 
def find_q_score(tfidf_dict_list, folder_path, normed_q_list):
  scores = tfidf_dict_list
  for dict in scores:
    for key in dict:
      for key2 in normed_q_list:
        if  key == key2 :
          dict[key] = dict[key] * normed_q_list[key2]
  score_list = []
  for dict in scores:
    doc_score = 0
    for key in dict:
      for key2 in normed_q_list:
        if key == key2:
          doc_score = doc_score + dict[key]
    score_list.append(doc_score)
  #relate files to scores
  file_dict = {}
  review_list=os.listdir(folder_path)

  for index in range(len(review_list)):
    #pathname = os.path.join(folder_path, review_list[index])
    #input_file = open(pathname, 'r')
    #doc = input_file.read()
    #d_l = doc.split('\n')
    #print "title: " + d_l[0]
    #print "description: " + d_l[1]
    title = review_list[index]#d_l[0]
    #input_file.close()
    file_dict[title] = score_list[index]
  #sort file scores
  items = sorted(file_dict.items(), key=get_count, reverse=True)
  use_cents = []
  items_cents = []
  aw = "The top twenty scoring books (highest to lowest) based on your query are: "
  print aw
  out_file = open("output.txt", "a")
  out_file.write(aw)

  num = 1
  for item in items[:20]:
    print str(num)+". ", "Title: ", item[0], " Score: ", item[1] #print results 
    out_file.write("\n" + str(num)+".  Title: "+ str(item[0]) )#+ " Score: "+ str(item[1])
    num = num + 1
    use_cents.append(item[0])
  out_file.close()
  #for item in items:
  #    print item[0]
  #    print item[1]
  items_cents.append(items)
  items_cents.append(use_cents)
  return items_cents
 
def get_query(q):
  q_wordcount = {}
  input_query = open(q, 'r')
  text = input_query.read()
  input_query.close()
  words = re.split('\W+', text)
  for word in words:
    word = word.lower()
    if len(word) >= 3:
      if word.isalpha():
        if not word in q_wordcount:
          q_wordcount[word] = 1
        else:
          q_wordcount[word] = q_wordcount[word] + 1
  return q_wordcount        
  

def part_1(folder_path, q):  
 tfidf_dict_list = find_norm(create_tfidf(folder_path))
 normed_q_list = norm_q(q)
 score_list_tf = find_doc_tf_score(tfidf_dict_list, folder_path)
 items_cen = find_q_score(tfidf_dict_list, folder_path, normed_q_list)
 score_list_q = items_cen[0]
 tf_q_cents_list = []
 tf_q_cents_list.append(score_list_tf)
 tf_q_cents_list.append(score_list_q)
 tf_q_cents_list.append(items_cen[1])
 return tf_q_cents_list

def get_centss(names_list, tf_items):
  cents = []
  for item in tf_items:
    for name in names_list:
      if item[0] == name:
        cents.append(item[1])
  return cents  


def cluster(score_list, cur):
  clusters = {}
  unused = []
  for item in score_list:
    #find which centroid is closest
    min_dist = 0
    centroid = 0
    for centroid in cur:
       test_dist = abs(item[1] - centroid)
       if min_dist == 0:
         min_dist = test_dist
         c_c = centroid
       else:        
         if test_dist < min_dist:
           min_dist = test_dist
           c_c = centroid
    #print c_c, item
    if clusters.has_key(c_c):
      clusters[c_c].append(item)
    else:
      clusters[c_c] = [item]
  for centroid in cur:
    if clusters.has_key(centroid):
      a = 1;
    else:
     
     unused.append(centroid)
  #print "unused: " + str(len(unused))
  return clusters
 
def update_cents(clusters):
  new = []
  clusters_ud = {}
  cluster_keys = sorted(clusters.keys())
  cluster_vals = []
  for key in clusters:
    cluster_v = []
    
    for item in clusters[key]: 
      cluster_v.append(item[1])
    cluster_vals.append(cluster_v)
  for list in cluster_vals:
    a = np.array(list)
    new.append(np.mean(a, axis = 0 ))
  return new 

  
def convergence(cur, old):
  ans = cmp(old, cur)
  #ans = (set([tuple(a) for a in cur_points]) == set([tuple(a) for a in old_points])
  return ans

def k_means(score_list, k, cent_list):
  #old = random.sample(score_list, k)
  #old = cent_list
  #cur = random.sample(score_list, k)
  #cur = cent_list
  cur_points = cent_list
  #for item in cur:
  #  cur_points.append(item[1])
  old_points = cent_list
  #for item in old:
  #  old_points.append(item)
  clusters = cluster(score_list, cur_points)
  cur_points = update_cents(clusters)
  here = 1
  while convergence(cur_points, old_points) != 0:
    #print "iteration: " + str(here)
    old_points = cur_points
    #assign items in score_list to clusters
    clusters = cluster(score_list, cur_points)
    #reevaluate centroids
    cur_points = update_cents(clusters)
    here = here + 1
    #print "OLD CENTS: " , old_points, "NEW CENTS", cur_points
    if convergence(cur_points, old_points) == 0:
      print "CONVERGENCE REACHED"
  return clusters

@app.route('/librecommend', methods=['GET', 'POST'])
def librecommend():
    if request.method == 'GET':
        return render_template('librecommend.html');
    elif request.method == 'POST':
        query = request.form['query']; 
        pathh = os.getcwd()
        endd = "\\books_small"
        folder_path =  pathh + endd; #pathh + endd #sys.argv[1]
        q_file = open("query.txt", "w")
        user_query = query #raw_input("PLEASE ENTER YOUR QUERY RELATED TO AUTHOR / TITLE / BOOK CONTENT: ")
        q_file.write(user_query)
        q_file.close()
        time.sleep(5)
        q = "query.txt"
        out_file = open("output.txt", "w")
        out_file.close()
        tf_q_cents_list = part_1(folder_path, q)
        #get random seeds
        #intitialize clusters with one point
        #clusters (K = 5)
        out_file = open("output.txt", "a")
        out_file.write("\n" + "The following selections are grouped based off the top results in respective order:")
        print "The following selections are grouped based off the top results in respective order:"
        cent_list = get_centss(tf_q_cents_list[2], tf_q_cents_list[0])
        clustt = k_means(tf_q_cents_list[0], 20, cent_list)
        place = 1
        for key in clustt:
         print "\n"+ str(len(clustt[key])) +" ITEMS IN GROUP FOR RESULT #"+str(place)+ " ARE: "
         out_file.write("\n"+ "\n" + str(len(clustt[key])) +" ITEMS IN GROUP FOR RESULT #"+str(place)+ " ARE: ")
         for item in clustt[key]:
           #print item[0]
           out_file.write("\n" + str(item[0]))
         place = place + 1
        out_file.close()
        f = open("output.txt", 'r')
        content = f.read()
        f.close()
        return redirect(url_for('results'));	 

@app.route('/how-to-use-me')
def how_to_use_me():
    # url1 = url_for('hello');
    # url2 = url_for('how_to_use_me');
    # url3 = url_for('list_of_authors');
    # url4 = url_for('about');
    # url5 = url_for('sources_cited');
    
    # link1 = '<a href="' + url1 + '">Home</a>';
    # link2 = '<a href="' + url2 + '">How to Use Me</a>';
    # link3 = '<a href="' + url3 + '">List of Authors</a>';
    # link4 = '<a href="' + url4 + '">About</a>';
    # link5 = '<a href="' + url5 + '">Source Cited</a>';

    return render_template('how-to-use-me.html');
	
	
@app.route('/results', methods=['GET','POST'])
def results():
    # url1 = url_for('hello');
    # url2 = url_for('how_to_use_me');
    # url3 = url_for('list_of_authors');
    # url4 = url_for('about');
    # url5 = url_for('sources_cited');
    
    # link1 = '<a href="' + url1 + '">Home</a>';
    # link2 = '<a href="' + url2 + '">How to Use Me</a>';
    # link3 = '<a href="' + url3 + '">List of Authors</a>';
    # link4 = '<a href="' + url4 + '">About</a>';
    # link5 = '<a href="' + url5 + '">Source Cited</a>';
    if request.method == 'GET':
      print "here at get"
      f = open("output.txt", 'r')
      content = f.read()
      f.close()
      return render_template('results.html', content = content.decode("latin1") );
    if request.method == 'POST':

      print "here at post"
      book = request.form['book'];
    #get textfile
      open_book = open('books_small/' + book);
      book_content = open_book.read()
      open_book.close()
      #print book_content
      b_c = book_content.decode("latin1");
      book_content = book_content.decode("latin1");
      #session['b_c'] = b_c
      resp = make_response(redirect(url_for('display_tag')) )     #render_template('display-book.html', book_content = book_content))
      resp.set_cookie('book_content', book_content);
      return resp#redirect(url_for('display_tag'))
      #return render_template('display-book.html', book_content = book_content.decode("latin1"));		

@app.route('/about')
def about():
    # url1 = url_for('hello');
    # url2 = url_for('how_to_use_me');
    # url3 = url_for('list_of_authors');
    # url4 = url_for('about');
    # url5 = url_for('sources_cited');
    
    # link1 = '<a href="' + url1 + '">Home</a>';
    # link2 = '<a href="' + url2 + '">How to Use Me</a>';
    # link3 = '<a href="' + url3 + '">List of Authors</a>';
    # link4 = '<a href="' + url4 + '">About</a>';
    # link5 = '<a href="' + url5 + '">Source Cited</a>';

    return render_template('about.html');

@app.route('/authors', methods=['GET','POST'])
def list_of_authors():
    # url1 = url_for('hello');
    # url2 = url_for('how_to_use_me');
    # url3 = url_for('list_of_authors');
    # url4 = url_for('about');
    # url5 = url_for('sources_cited');
    
    # link1 = '<a href="' + url1 + '">Home</a>';
    # link2 = '<a href="' + url2 + '">How to Use Me</a>';
    # link3 = '<a href="' + url3 + '">List of Authors</a>';
    # link4 = '<a href="' + url4 + '">About</a>';
    # link5 = '<a href="' + url5 + '">Source Cited</a>';
    if request.method == 'GET':
      print "here at get"
      f = open("authors.txt", 'r')
      content = f.read()
      f.close()
      return render_template('authors.html', content = content.decode("latin1") );
    #return render_template('authors.html');

@app.route('/sources-cited')
def sources_cited():
    # url1 = url_for('hello');
    # url2 = url_for('how_to_use_me');
    # url3 = url_for('list_of_authors');
    # url4 = url_for('about');
    # url5 = url_for('sources_cited');
    
    # link1 = '<a href="' + url1 + '">Home</a>';
    # link2 = '<a href="' + url2 + '">How to Use Me</a>';
    # link3 = '<a href="' + url3 + '">List of Authors</a>';
    # link4 = '<a href="' + url4 + '">About</a>';
    # link5 = '<a href="' + url5 + '">Source Cited</a>';

    return render_template('sources-cited.html');


@app.route('/display-book', methods=['GET','POST'])
def display_tag():
    # url1 = url_for('hello');
    # url2 = url_for('how_to_use_me');
    # url3 = url_for('list_of_authors');
    # url4 = url_for('about');
    # url5 = url_for('sources_cited');
    # link1 = '<a href="' + url1 + '">Home</a>';
    # link2 = '<a href="' + url2 + '">How to Use Me</a>';
    # link3 = '<a href="' + url3 + '">List of Authors</a>';
    # link4 = '<a href="' + url4 + '">About</a>';
    # link5 = '<a href="' + url5 + '">Source Cited</a>';
    #b_c = request.args['b_c']  # counterpart for url_for()
    #b_c = session['b_c']       # counterpart for session
    #username = request.cookies.get('username')
    book_content = request.cookies.get('book_content');
    if request.method == 'GET':
      print "here at get"
      return render_template('display-book.html', book_content = book_content );
    if request.method == 'POST':

      print "here at post"
      tagg = "TAG: " + request.form['tagg'];
    #get textfile
      open_book = open('books_small/' + book , "a");
      open_book.write(tagg)
      open_book.close()
      #return render_template('display-book.html', book_content = book_content.decode("latin1"));

# @app.route('/question/<title>', methods=['GET', 'POST'])
# def question(title):
#     if request.method == 'GET':
#         # Redis code to load question
#         question = r.get(title+':question')
#         return render_template('AnswerQuestion.html',
#                                question = question)
#     elif request.method == 'POST':
#         submittedAnswer = request.form['submittedAnswer'];

#         # Redis code to load answer
#         answer=r.get(title+':answer')

#         if submittedAnswer == answer:
#             return render_template('Correct.html');
#         else:
#             return render_template('Incorrect.html',
#                                    answer = answer,
#                                    submittedAnswer = submittedAnswer);

# @app.route('/submit', methods=['GET', 'POST'])
# def submit():
#     if request.method == 'GET':
#         return render_template('CreateQuestion.html');
#     elif request.method == 'POST':
#         question = request.form['question'];
#         answer = request.form['answer'];
#         title = request.form['title'];

#         # Redis code to save question and answer
#         r.set(title+':question',question);
#         r.set(title+':answer',answer);      

#         return render_template('CreatedQuestion.html',
#                                question = question);
#     return;