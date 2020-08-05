# -----------------------------------------------------------------PORTRAY----------------------------------------------------------------------#
# ----------------------------------------------------------------IMAGE ANALYSER----------------------------------------------------------------#

'''DISABLE WARNINGS'''

import os
# os.chdir('C:\\Users\\infin\\Anaconda3\\envs\\FlipkartProject\\Projects\\APP')
import warnings
import gc
import sys
import yake
import urllib.request
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import time 
import json
import glob
import pickle
import random
from pathlib import Path
import pickle
import cv2
import editdistance
import string
from sklearn.preprocessing import MinMaxScaler
import io
import itertools
import networkx as nx
import nltk
import re
import networkx
from rake_nltk import Rake
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
from imgaug import augmenters as iaa
from sklearn.model_selection import StratifiedKFold, KFold
import mrcnn
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
import keras.layers
from mrcnn.model import log
from mrcnn.model import log, BatchNorm
COCO_WEIGHTS_PATH = 'mask_rcnn_coco.h5'
warnings.filterwarnings("ignore", category=DeprecationWarning) 


'''SET CONFIGURATIONS FOR MODEL'''

NUM_CATS = 46
IMAGE_SIZE = 512
class FashionConfig(Config):
    NAME = "fashion"
    NUM_CLASSES = NUM_CATS + 1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4
    BACKBONE = 'resnet50'
    IMAGE_MIN_DIM = IMAGE_SIZE
    IMAGE_MAX_DIM = IMAGE_SIZE    
    IMAGE_RESIZE_MODE = 'none'
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 200  
config = FashionConfig()


'''LOAD LABELS FOR IMAGE SEGMENTATION'''

with open("label_descriptions.json") as f:
    label_descriptions = json.load(f)
label_names = [x['name'] for x in label_descriptions['categories']]


'''Helper Functions For Image Analysis'''

def resize_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)  
    return img

def to_rle(bits):
    rle = []
    pos = 0
    for bit, group in itertools.groupby(bits):
        group_list = list(group)
        if bit:
            rle.extend([pos, sum(group_list)])
        pos += len(group_list)
    return rle

def refine_masks(masks, rois):
    areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0)
    mask_index = np.argsort(areas)
    union_mask = np.zeros(masks.shape[:-1], dtype=bool)
    for m in mask_index:
        masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))
        union_mask = np.logical_or(masks[:, :, m], union_mask)
    for m in range(masks.shape[-1]):
        mask_pos = np.where(masks[:, :, m]==True)
        if np.any(mask_pos):
            y1, x1 = np.min(mask_pos, axis=1)
            y2, x2 = np.max(mask_pos, axis=1)
            rois[m, :] = [y1, x1, y2, x2]
    return masks, rois

augmentation = iaa.Sequential([
    iaa.Fliplr(0.5) # only horizontal flip here
])


'''Model Setup And Download'''

model_path = 'mask_rcnn_fashion_0008.h5'
class InferenceConfig(FashionConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()
model = modellib.MaskRCNN(mode='inference', 
                          config=inference_config,
                          model_dir='../Mask_RCNN/')

assert model_path != '', "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

print("MODEL LOADED")
print()


'''Main Functions for Image Analysis -> Download, Save and Run Predictions'''

def main(df):
    
    feature_list = []
    missing_count = 0
    
    os.chdir('static/Images/')
    for i in range(len(df)):
        image_url = df["Image_Link"][i]
        save_name = df["Name"][i] + '.jpg'
        urllib.request.urlretrieve(image_url, save_name)

    for i in tqdm(range(len(df))):
        labels = []
        path = df["Name"][i] + '.jpg'
        try:
            image = resize_image(path)
            result = model.detect([image])[0]
        except:
            print(df["Name"][i])
            feature_list.append([1])
            continue
    
        if result['masks'].size > 0:
            masks, _ = refine_masks(result['masks'], result['rois'])
            for m in range(masks.shape[-1]):
                mask = masks[:, :, m].ravel(order='F')
                rle = to_rle(mask)
                label = result['class_ids'][m] - 1
                labels.append(label)
            feature_list.append(list(set(labels)))
        else:
            feature_list.append([1])
            missing_count += 1
            
    for i in range(len(feature_list)):
        for j in range(len(feature_list[i])):
            feature_list[i][j] = label_names[feature_list[i][j]]
            
    df["Feature"] = pd.Series(feature_list)
    os.chdir('..')
    os.chdir('..')
    return df

def getanalysis(df):
    lis = []
    language = "en"
    max_ngram_size = 1
    deduplication_thresold = 0.9
    deduplication_algo = 'seqm'
    windowSize = 1
    numOfKeywords = 2

    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold,    dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
    for i in df["Description"]:
        if (i == ''):
            continue
        keywords = custom_kw_extractor.extract_keywords(i)
        temp = []
        for j in keywords:
            temp.append(j[0])
        lis.append(temp)
            
    return lis

def cleanresults(df):
    del df["Discount"], df["Rating"], df["Number of Ratings"], df["Reviews"], df["Current Views"]
    lis = getanalysis(df)
    df["Keywords"] = pd.Series(lis)
    return df


print("SETUP COMPLETE")
print()


# --------------------------------------------------------------DATA SCRAPER-------------------------------------------------------------------#


options = webdriver.ChromeOptions()
options.add_argument('start-maximized') 
options.add_argument('disable-infobars')
options.add_argument('--disable-extensions')

class DataCollectionEcomm:

    def __init__(self, base_site, search, path, query = ['T-Shirt']):
        self.browser = self.genrateBroswer()
        self.links = []
        self.base_site = base_site
        self.path = path
        self.search = search
        self.query = query
        self.df = pd.DataFrame(columns=["Name", "Brand", "Price", "Discount", "Image_Link", "Rating", "Number of Ratings", "Reviews", "Current Views", "Description"])
        
    def getalllinkstoproduct(self, query):
        self.browser.find_element_by_xpath(self.search["search_box"]).click()
        self.browser.implicitly_wait(5)
        self.browser.find_element_by_xpath(self.search["search_input"]).send_keys(query)
        self.browser.implicitly_wait(10)
        self.browser.find_element_by_xpath(self.search["search_input"]).send_keys(Keys.ENTER)
        temps = []
        for i in range(1,1000):
            lis =self.browser.find_elements_by_css_selector(self.search["product_selector"] + str(i) + self.search["product_selector_no"])
            if (not lis):
                break
            temps.append(lis[0].get_attribute('href'))
        self.browser.get(self.base_site)
        self.browser.implicitly_wait(5)
        return temps   
        
    def genrateBroswer(self):
        self.browser = webdriver.Chrome(options=options)
        return self.browser

    def getproductdata(self):
        self.browser.implicitly_wait(3)
        Product_Name = self.browser.find_element_by_xpath(self.path["p_name"]).text
        try:
            Product_Brand = self.browser.find_element_by_xpath(self.path["p_brand"]).text
        except:
            Product_Brand = Product_Name
            
        try:
            Product_Price = self.browser.find_element_by_xpath(self.path["p_price"]).text
        except:
            Product_Price = "Out Of Stock"

        try:
            Product_Disc = self.browser.find_element_by_xpath(self.path["p_disc"]).text[:3]
            print(1)
        except:
            Product_Disc = 'NULL'
          
        try:
            Product_Image = self.browser.find_element_by_xpath(self.path["p_img"]).get_attribute("src")
        except:
            Product_Image = self.browser.find_element_by_xpath(self.path["p_img2"]).get_attribute("src")
            
        for second in range(0,50):
            self.browser.execute_script("window.scrollBy(0,300)", "")
            time.sleep(5) 
            try:
                self.browser.find_element_by_id(self.path["p_rev"])
                break
            except:
                continue
        
        Product_Reviews = []
        
        try:
            Product_Rating = self.browser.find_element_by_xpath(self.path["p_rat"]).text
        except:
            Product_Rating = "None"
            print("Help - STOP")
            
        try:
            Product_NumRatings = self.browser.find_element_by_xpath(self.path["p_numrat"]).text
        except:
            Product_NumRatings = "Zero"
            print("Help - STOP")
            
        try:
            Curr_Views = self.browser.find_element_by_xpath(self.path["p_curr"]).text
        except:
            Curr_Views = "0"
            print('Help')
            
        try:
            Product_Desc = self.browser.find_element_by_xpath("//*[@id='product-page-selling-statement']").text
        except:
                Product_Desc = ""
                print("Help")
        
        reviews = self.browser.find_elements_by_class_name("_2k-Kq")
        for x in reviews:
            subject = x.find_element_by_class_name("_3P2YP").text
            text = x.find_element_by_class_name("_2wSBV").text
            stars = x.find_element_by_class_name("_3tZR1").value_of_css_property('width')[:-2]
            Product_Reviews.append([subject, text, stars])
        
        self.df = self.df.append({'Name': Product_Name, 'Brand': Product_Brand, "Price": Product_Price, "Discount": Product_Disc, "Image_Link": Product_Image, "Rating": Product_Rating, "Number of Ratings": Product_NumRatings, "Reviews": Product_Reviews, "Current Views": Curr_Views, "Description": Product_Desc}, ignore_index=True)
        
    def helper(self, link):
        self.browser.get(link)
        
    def main_1(self):
        self.browser.get(self.base_site)
        self.browser.delete_all_cookies()
        temp = []
        time.sleep(10)
        for i in self.query:
            link = self.getalllinkstoproduct(i)
            temp += link
    
        link_set = set(temp)
        self.links = list(link_set)
        return self.links
    
    def main_2(self):
        for i in tqdm(range(len(self.links))):
            self.helper(self.links[i])
            time.sleep(5)
            self.getproductdata()
            
            
'''FOR SHEIN'''

# 1. Comment out:
#         for second in range(0,50):
#             self.browser.execute_script("window.scrollBy(0,300)", "")
#             time.sleep(5) 
#             try:
#                 self.browser.find_element_by_id(self.path["p_rev"])
#                 break
#             except:
#                 continue

# 2. Change:
# self.browser.find_element_by_xpath(self.search["search_input"]).send_keys(query)
# self.browser.implicitly_wait(10)
# self.browser.find_element_by_xpath(self.search["search_input"]).send_keys(Keys.ENTER)

# 3. Change:
# reviews = self.browser.find_elements_by_class_name('common-reviews__list-item-detail')
# for i in range(len(reviews)):
#     subject = ''
#     text = reviews[i].find_element_by_class_name("rate-des").text
#     stars = Product_Rating
#     Product_Reviews.append([subject, text, stars])


# --------------------------------------------------------------Review Weights-------------------------------------------------------------------#


class WeightingReviews:
    
    def __init__(self, df):
        self.df = df
        self.k = 0.3
        
    def setup_environment(self):
        """Download required resources."""
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        print('Completed resource downloads.')
        
    def filter_for_tags(self, tagged, tags=['NN', 'JJ', 'NNP']):
        """Semantic Filter Based on POS."""
        return [item for item in tagged if item[1] in tags]


    def normal(self, tagged):
        return [(item[0].replace('.', ' '), item[1]) for item in tagged]


    def unique_ever(self, iterable, key=None):
    
        seen = set()
        seen_add = seen.add
        if key is None:
            for element in [x for x in iterable if x not in seen]:
                seen_add(element)
                yield element
        else:
            for element in iterable:
                k = key(element)
                if k not in seen:
                    seen_add(k)
                    yield element


    def build_graph(self, nodes):
        """Return a networkx graph instance.
        :param nodes: List of hashables that represent the nodes of a graph.
        """
        gr = nx.Graph()  
        gr.add_nodes_from(nodes)
        nodePairs = list(itertools.combinations(nodes, 2))

        for pair in nodePairs:
            firstString = pair[0]
            secondString = pair[1]
            levDistance = editdistance.eval(firstString, secondString)
            gr.add_edge(firstString, secondString, weight=levDistance)

        return gr


    def extract_key_phrases(self, text):
        word_tokens = nltk.word_tokenize(text)

        tagged = nltk.pos_tag(word_tokens)
        textlist = [x[0] for x in tagged]

        tagged = self.filter_for_tags(tagged)
        tagged = self.normal(tagged)

        unique_word_set = self.unique_ever([x[0] for x in tagged])
        word_set_list = list(unique_word_set)

        graph = self.build_graph(word_set_list)

        calculated_page_rank = nx.pagerank(graph, weight='weight')
        keyphrases = sorted(calculated_page_rank, key=calculated_page_rank.get,
                        reverse=True)
        one_third = 50
        keyphrases = keyphrases[0:50]

        modified_key_phrases = set([])
        dealt_with = set([])
        i = 0
        j = 1
        while j < len(textlist):
            first = textlist[i]
            second = textlist[j]
            if first in keyphrases and second in keyphrases:
                keyphrase = first + ' ' + second
                modified_key_phrases.add(keyphrase)
                dealt_with.add(first)
                dealt_with.add(second)
            else:
                if first in keyphrases and first not in dealt_with:
                    modified_key_phrases.add(first)
                if j == len(textlist) - 1 and second in keyphrases and \
                        second not in dealt_with:
                    modified_key_phrases.add(second)

            i = i + 1
            j = j + 1

        return modified_key_phrases
    
    def raking(self, text):
        r = Rake(min_length=1, max_length=3)
        r.extract_keywords_from_text(text)
        ans = r.get_ranked_phrases_with_scores()
        return ans
    
    def calcweight(self, text, final):
        count = 0
        words = word_tokenize(text)
        for i in words:
            if i in final:
                count += 1
        weight = (count/len(final)) * 100
        return weight
    
    def main_weights(self):
        text = ""
        for i in self.df["Reviews"]:
            for j in i:
                text = text + "" + j
        
        pattern = '[0-9]'
        text = re.sub(pattern, ' ', text)

        result_rake = self.raking(text)
        final = []
        for i in result_rake:
            if (i[0] > 8):
                lis = nltk.word_tokenize(i[1])
                final += lis
        result_textrank = self.extract_key_phrases(text)
        final += result_textrank


        resulting = []

        for i in self.df["Reviews"]:
            lis = []
            if (not i):
                lis.append(self.k)
                resulting.append(lis)
                continue
            for text, score in i.items():
                weight_factor = self.calcweight(text, final)
                a = weight_factor + self.k
                lis.append(a)
            resulting.append(lis)
    
        self.df["Weights"] = pd.Series(resulting)
        return self.df
    
    
# --------------------------------------------------------------Pre Processor-------------------------------------------------------------------#
 
    
class PreProcessEcomm:
    
    def __init__(self, df):
        self.df = df
        
    def simplify(self, rev):
        reviews = []
        if (type(rev) != str):
            for i in rev:
                text = i[0] + i[1] + ' '  + i[2]
                reviews.append(text)
            return reviews
            
        temp = rev.split(']')
        reviews = []
        for i in temp:
            if i != '':
                reviews.append(i)
        return reviews


    def clean(self, rev): 
        lis = []
        for i in rev:
            i = re.sub(r'[^\w\s]','',i)
            i = i.replace("\n", " ")
            lis.append(i)
        return lis

    def clean2(self, rev):
        try:
            i = re.sub(r'[^\w\s]','',rev)
            i = i.replace("\n", " ")
            return i
        except:
            return ""
        
    def reviewtodict(self):
        lis = []
        for i in self.df["Reviews"]:
            a = {}
            for j in i:
                try:
                    score = int(j[-2:])
                    text = j[:len(j) - 2]
                    a[text] = score
                except:
                    score = 0
                    text = j[:len(j) - 2]
                    a[text] = score
            lis.append(a)
        self.df["Reviews"] = pd.Series(lis)
        return
    
    def ratings(self, s):
        x = s[:3]
        try:
            return float(x)
        except:
            return 0
    
    def num_ratings(self, s):
        try:
            x = re.findall(r'\d+', s) 
            return int(x[0])
        except:
            return 0

    def curr_views(self, s):
        try:
            x = re.findall(r'\d+', s)[0]
            ans = int(x)
            return ans
        except:
            return 0

    def price(self, s):
        try:
            x = re.findall('[\$\£\€](\d+(?:\.\d{1,2})?)', s)
            return float(x[0])
        except:
            s = s[1:]
            return float(s[:4])

    def discount(self, s):
        if (s == 0):
            return 0
        elif (s == None):
            return 0
        else:
            return int(re.findall(r'\d+', s)[0])
        
    def main_pre(self):
        self.df['Reviews']= self.df.Reviews.apply(self.simplify)
        self.df['Reviews']= self.df.Reviews.apply(self.clean)
        self.df['Discount'] = self.df['Discount'].fillna(0)
        self.df["Rating"] = self.df["Rating"].apply(self.ratings)
        self.df["Number of Ratings"] = self.df["Number of Ratings"].apply(self.num_ratings)
        self.df["Current Views"] = self.df["Current Views"].apply(self.curr_views)
        self.df["Price"] = self.df["Price"].apply(self.price)
        self.df["Discount"] = self.df["Discount"].apply(self.discount)
        self.df['Description'] = self.df.Description.apply(self.clean2)
        
        self.reviewtodict()
        
        return self.df
    
    
# --------------------------------------------------------------PORTRAY - ECOMMERCE--------------------------------------------------------------# 
    

scaler = MinMaxScaler(feature_range = (0,10))
class PORTRAY_E:
    
    def __init__(self, df):
        self.df = df
        
    def normalize(self):
        column_names_to_normalize = ['RSCORE']
        x = self.df[column_names_to_normalize].values
        x_scaled = scaler.fit_transform(x)
        df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = self.df.index)
        self.df[column_names_to_normalize] = df_temp
    
    def r1score(self):
        mean_revs = self.df["Number of Ratings"].mean()
        mean_views = self.df["Current Views"].mean()
    
        r1scores = []
    
        for i in range(len(self.df)):
            rating = self.df["Rating"][i]
            views = self.df["Current Views"][i]
            count = self.df["Current Views"][i]
        
            factor = (views + count) / 2
            r1 = factor*rating
            r1scores.append(r1)
        
        self.df["R1SCORE"] = pd.Series(r1scores)
        mean_dist = self.df["R1SCORE"].mean()
        self.df["R1SCORE"] = self.df["R1SCORE"] / mean_dist
        return

    def r2score(self):
    
        r2scores = []
    
        for i in range(len(self.df)):
            currdict = self.df["Reviews"][i]
            weights = self.df["Weights"][i]
            if (not currdict):
                r2scores.append(weights[0])
                continue
            j = 0
            r2 = 0
            for key, val in currdict.items():
                r2 = r2 + (val*weights[j])
                j += 1
            r2scores.append(r2/10)
    
        self.df["R2SCORE"] = pd.Series(r2scores)
        return

    def rscore(self):
    
        rscores = []
        for i in range(len(self.df)):
            r = (self.df["R1SCORE"][i] + self.df["R2SCORE"][i]) / 2
            rscores.append(r)
    
        self.df["RSCORE"] = pd.Series(rscores)
        del self.df["R1SCORE"], self.df["R2SCORE"]
        del self.df["Weights"]
        return
    
    def price_discount(self):
        P_mean = self.df["Price"].mean()
        total = 0
        count = 0
        for i in self.df["Discount"]:
            if i != 0:
                total = total + i
                count += 1
        if count == 0:
            D_mean = 0
        else:
            D_mean = total/count
    
        lis = []
        dis = []
        for i in range(len(self.df)):
            if (self.df["Price"][i] >= 2*P_mean and self.df["RSCORE"][i] > 5.00):
                self.df["RSCORE"][i] + 0.5
            elif (self.df["Price"][i] <= 0.5*P_mean and self.df["RSCORE"][i] < 3.00):
                self.df["RSCORE"][i] - 0.5
        for i in range(len(self.df)):
            if (self.df["Discount"][i] >= 1.5*D_mean and self.df["RSCORE"][i] < 5.00):
                self.df["RSCORE"][i] - 0.5
                
    def results(self, n = 5, m = 5):
        
        self.r1score()
        self.r2score()
        self.rscore()
        self.price_discount()
        self.normalize()
        
        self.df = self.df.sort_values(by='RSCORE', ascending = False)
        self.df = self.df.reset_index(drop = True)
        TOP_PRO = self.df[:n]
        TOP_PRO = TOP_PRO.reset_index(drop = True)
        BOT_PRO = self.df[-m:]
        BOT_PRO = BOT_PRO.reset_index(drop = True)   
        return TOP_PRO, BOT_PRO


# --------------------------------------------------------------CORE CODE--------------------------------------------------------------# 
    
    
def predictor(choice, query):
    m = {}
    m['NORDSTROM'] = 'NORDSTROM'
    m['SHEIN'] = 'SHEIN'

    choice = [choice]
    file = ''
    print()

    if (len(choice) == 1):
        file = m[choice[0]]
        util = pickle.load(open('static/PKL/' + file + '.pkl', "rb"))
        util[1] = [query]
        for i in util[1]:
            try:
                df = pd.read_csv('static/CSV/' + file + '_' + i + '.csv')
                break
            except:
                print('WE WILL HAVE TO SCRAPE THIS DATA.')
                print()
                scraper = DataCollectionEcomm(util[0], util[2], util[3], util[1])
                links = scraper.main_1()
                scraper.main_2()
                df = scraper.df
                df.to_csv('static/CSV/' + file + '_' + i + '.csv', index = False)
                break
        pre = PreProcessEcomm(df)
        df = pre.main_pre()
        wgt = WeightingReviews(df)
        df = wgt.main_weights()
        alg = PORTRAY_E(df)
        top, bottom = alg.results(5,5)
    
    else:
        queries = [item for item in input("Enter the Queries: ").split()] 
        DATA = pd.DataFrame(columns=["Name", "Brand", "Price", "Discount", "Image_Link", "Rating", "Number of Ratings", "Reviews", "Current Views", "Description"])
        for i in choice:
            file = m[i]
            util = pickle.load(open('static/PKL/' + file + '.pkl', "rb"))
            util[1] = queries
            for i in util[1]:
                try:
                    df = pd.read_csv('static/CSV/' + file + '_' + i + '.csv')
                    DATA.append(df)
                    reak
                except:
                    scraper = DataCollectionEcomm(util[0], util[2], util[3], util[1])
                    links = scraper.main_1()
                    scraper.main_2()
                    df = scraper.df
                    DATA.append(df)
                    df.to_csv('static/CSV/' + file + '_' + i + '.csv', index = 'False')
                    break
            pre = PreProcessEcomm(df)
            df = pre.main_pre()
            DATA.append(df)
        wgt = WeightingReviews(DATA)
        df = wgt.main_weights()
        alg = PORTRAY_E(DATA)
        top, bottom = alg.results(5,5)
    
    df_top = main(top)
    df_bottom = main(bottom)

    df_top = cleanresults(df_top)
    df_bottom = cleanresults(df_bottom)
    
    df_top.to_pickle('static/Sample_Results/TOP.pkl')
    df_bottom.to_pickle('static/Sample_Results/BOTTOM.pkl')

    print('Results are saved')
    print('Terminating')


'''--------------------------------------------------------------------E N D-----------------------------------------------------------------'''
