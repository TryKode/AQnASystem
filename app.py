# 1. Importing Libraries
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from bs4 import BeautifulSoup as bs
import requests
import re

from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import torch
import textwrap

from textblob import TextBlob

# 2. Create Class for taking Data in
class database_qna(BaseModel):
    context : str 
    question : str

class database_scrape(BaseModel):
    product_url: str 

# 3. Create the app and model objects
app = FastAPI()
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# 4. API Base page
@app.get('/')
def index():
    return {'message': 'Hello, This API is used to Scrape the data from Amazon Product page and to use BERT-Transformer for question-answering system.'}

# 5. API to Scrape data
@app.get('/scrape')
def scrape_data(database_scrape:database_scrape):
    productURL = database_scrape.product_url
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36", "Accept-Encoding":"gzip, deflate", "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "DNT":"1","Connection":"close", "Upgrade-Insecure-Requests":"1"}
    productPage = requests.get(productURL, headers=headers)
    productSoup = bs(productPage.content,'html.parser')

    productNames = productSoup.find_all('span', id='productTitle')
    productNames = productNames[0].get_text().strip()
    
    ids = ['priceblock_dealprice', 'priceblock_ourprice', 'tp_price_block_total_price_ww', 'apexPriceToPay']
    for ID in ids:
        productDiscountPrice = productSoup.find_all('span', id=ID)
        if len(productDiscountPrice) > 0 :
            break
    productDiscountPrice = productDiscountPrice[0].get_text().strip()
    productDiscountPrice = 'Product Price after Discount '+productDiscountPrice

    classes = ['priceBlockStrikePriceString', 'a-text-price']
    for CLASS in classes:
        productActualPrice = productSoup.find_all('span', class_=CLASS)
        if productActualPrice != [] :
            break
    productActualPrice = productActualPrice[0].get_text().strip()
    productActualPrice = 'Product Actual Price '+productActualPrice

    ###
    productRating = productSoup.find_all('span', class_="a-icon-alt")
    productRating = productRating[0].get_text().strip()
    ###

    productFeatures = productSoup.find_all('div', id='feature-bullets')
    productFeatures = productFeatures[0].get_text().strip()
    productFeatures = re.split('\n|  ',productFeatures)
    temp = []
    for i in range(len(productFeatures)):
        if productFeatures[i]!='' and productFeatures[i]!=' ' :
            temp.append( productFeatures[i].strip() )
    productFeatures = temp
    
    productSpecs = productSoup.find_all('table', id='productDetails_techSpec_section_1')
    productSpecs = productSpecs[0].get_text().strip()
    productSpecs = re.split('\n|\u200e|  ',productSpecs) 
    temp = []
    for i in range(len(productSpecs)):
        if productSpecs[i]!='' and productSpecs[i]!=' ' :
            temp.append( productSpecs[i].strip() )
    productSpecs = temp

    productDetails = productSoup.find_all('div', id='productDetails_db_sections')
    productDetails = productDetails[0].get_text()
    productDetails = re.split('\n|  ',productDetails) 
    temp = []
    for i in range(len(productDetails)):
        if productDetails[i]!='' and productDetails[i]!=' ' :
            temp.append( productDetails[i].strip() )
    productDetails = temp
    
    context = productNames + '\n' + productDiscountPrice + '. ' + productActualPrice + '.\n' + productRating + '.\n'
    i = 0
    while i<len(productFeatures):
        context = context + productFeatures[i]+', '
        i = i+1

    i = 0
    while i<len(productSpecs):
        context = context + productSpecs[i]+' '+productSpecs[i+1]+', '
        i = i+2
    context = context[:len(context)-2] + '.\n'

    # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> | ', productNames, productDiscountPrice, productActualPrice, productFeatures, productSpecs, productDetails, context, sep="_-_-_-_-_")
    details = {
        'product_data' : {
            'productNames' : productNames,
            'productDiscountPrice' : productDiscountPrice,
            'productActualPrice' : productActualPrice,
            'productRating' : productRating,
            'productFeatures' : productFeatures,
            'productSpecs' : productSpecs,
            'productDetails' : productDetails,
            'context' : context
        }
    }

    return details

# 6. API for Question-Answering System
@app.get('/qna')
async def qna_bert(database_qna : database_qna):
    def check_spelling(question):
        question = re.sub(r'[^\w\s]', '', question)
        question = question.lower()
        question_list = question.split()

        for i in range(len(question_list)):
            question_list[i] = str( TextBlob(question_list[i]).correct() )
        
        question = " ".join(question_list)
        return (question + " ?")

    def answer_question(question, answer_text):
        input_ids = tokenizer.encode(question, answer_text)
        sep_index = input_ids.index(tokenizer.sep_token_id)
        num_seg_a = sep_index + 1
        num_seg_b = len(input_ids) - num_seg_a

        segment_ids = [0]*num_seg_a + [1]*num_seg_b
        assert len(segment_ids) == len(input_ids)

        outputs = model(torch.tensor([input_ids]),token_type_ids=torch.tensor([segment_ids]),return_dict=True) 

        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores)
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        answer = tokens[answer_start]
        for i in range(answer_start + 1, answer_end + 1):
            if tokens[i][0:2] == '##':
                answer += tokens[i][2:]
            else:
                answer += ' ' + tokens[i]

        # print('Answer: "' + answer + '"')
        return answer

    context = database_qna.context
    question = check_spelling(database_qna.question)
    answer = answer_question(question, context)

    return {'context': context, 'question' : question, 'answer' : answer}

# 7. Run the API with uvicorn in Local system
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload