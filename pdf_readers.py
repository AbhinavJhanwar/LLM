from llm_models import *

# %%
from PyPDF2 import PdfReader
import tabula
import camelot
def get_pdf_text_pypdf2(pdf_docs, table_reader='Camelot'):
    parsed_data = []
    for pdf in pdf_docs:
        text = ""
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

        # handle tables
        if table_reader=='Camelot':
            tables = camelot.read_pdf(pdf, pages="all", flavor='lattice')
            table_markdowns = [table.df.to_markdown(index=False) for table in tables]
        else:
            tables = tabula.read_pdf(pdf, pages="all", multiple_tables=True)
            table_markdowns = [table.to_markdown(index=False) for table in tables]

        parsed_data.append({
                "text": text.strip(),
                "tables": table_markdowns
            })
    return parsed_data

# text = get_pdf_text_pypdf2(['sample pdfs/ContentStandards_StyleGuide_VideoGaming_01.29.25.pdf'])


# %%
import pymupdf4llm
import tabula
import camelot

# https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/api.html
def get_pdf_text_pymupdf4llm(pdf_docs, table_reader='tabula', image_reader=''):
    parsed_data = []
    if os.path.exists('pymupdf_images'):
        shutil.rmtree('pymupdf_images')
    for pdf in pdf_docs:
        content = pymupdf4llm.to_markdown(pdf, 
                                write_images=True, # Each image or vector graphic on the page will be extracted and stored as an image named "input.pdf-pno-index.extension" in a folder of your choice. 
                                dpi=500,
                                force_text=False,
                                show_progress=True,
                                image_path='pymupdf_images',
                                embed_images=False,
                                # page_chunks=True # get meta data of image
                                # table_strategy= "lines", # “lines”, “lines_strict” and “text”.
                                )

        # handle tables
        if table_reader=='Camelot':
            tables = camelot.read_pdf(pdf, pages="all", flavor='lattice')
            table_markdowns = [table.df.to_markdown(index=False) for table in tables]
        else:
            tables = tabula.read_pdf(pdf, pages="all", multiple_tables=True)
            table_markdowns = [table.to_markdown(index=False) for table in tables]

        # handle images
        if image_reader == 'Ollama':
            for root, dirs, files in os.walk('pymupdf_images'):
                for file in files:
                    # Construct full file path
                    file_path = os.path.join(root, file)
                    image_text = get_ollama_image_response(file_path)
                    # replace image path with image text in pdf text
                    content = content.replace(file_path, image_text)

        elif image_reader == 'OpenAI':
            for root, dirs, files in os.walk('pymupdf_images'):
                for file in files[:5]:
                    # Construct full file path
                    file_path = os.path.join(root, file)
                    image_text = get_chat_gpt_response(file_path)
                    # replace image path with image text in pdf text
                    content = content.replace(file_path, image_text)

        elif image_reader == 'Groq':
            for root, dirs, files in os.walk('pymupdf_images'):
                for file in files:
                    # Construct full file path
                    file_path = os.path.join(root, file)
                    image_text = get_groq_response(file_path)
                    # replace image path with image text in pdf text
                    content = content.replace(file_path, image_text)

        parsed_data.append({
                "text": content.strip(),
                "tables": table_markdowns
            })

    return parsed_data

# text = get_pdf_text_pymupdf4llm(['sample pdfs/1506.01497v3.pdf'])[0]['text']
# with open('example.txt', 'w') as file:
#         file.write(text)


# %%
import os
import shutil
import time
import re
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium import webdriver
from PIL import Image
import pytesseract
from docx import Document
from fpdf import FPDF

# https://www.amazon.com/Arnotts-Original-Value-Pack-330g/dp/B00DTDPUUU

# Function to capture screenshots of each page
def capture_screenshots(url, output_folder):

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.mkdir(output_folder)
    chrome_options = Options()
    chrome_options.binary_location = '/usr/bin/google-chrome'
    chrome_options.add_argument('--headless')  # Run Chrome in headless mode
    chrome_options.add_argument('--no-sandbox')  # Required for running as root user
    chrome_options.add_argument('--disable-dev-shm-usage')  # Required for running in Docker

    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)

    # Wait until page is fully loaded
    WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))

    # Scroll down to capture the entire page
    total_height = int(driver.execute_script("return document.body.scrollHeight"))
    screenshot_count = 0
    for i in range(0, total_height, 250):
        driver.execute_script("window.scrollTo(0, {});".format(i))
        time.sleep(0.2)  # Adjust the delay as needed

        # Capture screenshot
        screenshot_path = f"{output_folder}/screenshot_{screenshot_count + 1}.png"
        driver.save_screenshot(screenshot_path)

        # Open the screenshot image
        image = Image.open(screenshot_path)
        # Resize the image to a width of 1024 pixels
        image = image.resize((1024, int(image.height * 1024 / image.width)))
        # Save the resized image to a file
        image.save(screenshot_path)


        screenshot_count += 1

    driver.quit()
    return screenshot_count


def get_html_text(url='https://www.amazon.com/Arnotts-Original-Value-Pack-330g/dp/B00DTDPUUU'):
    output_folder='images'
    # n_pages = capture_screenshots(url, output_folder)
    # print(n_pages, url)

    text=""
    screenshots = [f for f in os.listdir(output_folder) if f.endswith('.png')]
    screenshots.sort(key=lambda x: int(re.search(r'\d+', x).group()))  # Sort numerically

    for screenshot in screenshots:
        image_path = os.path.join(output_folder, screenshot)
        text += pytesseract.image_to_string(Image.open(image_path))

    return text

# text = get_html_text()



# %%
import pymupdf
import tabula
import camelot

def get_pdf_text_pymupdf(pdf_docs, table_reader):
    parsed_data = []
    for pdf in pdf_docs:
        doc = pymupdf.open(pdf)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text("text")

        # handle tables
        if table_reader=='Camelot':
            tables = camelot.read_pdf(pdf, pages="all", flavor='lattice')
            table_markdowns = [table.df.to_markdown(index=False) for table in tables]
        else:
            tables = tabula.read_pdf(pdf, pages="all", multiple_tables=True)
            table_markdowns = [table.to_markdown(index=False) for table in tables]

        parsed_data.append({
            "text": text.strip(),
            "tables": table_markdowns
        })
        
    return parsed_data

# %%
def read_pdf(pdf_docs, pdf_reader, table_reader, image_reader):
    if pdf_reader == 'pytesseract':
        data = get_html_text()

    elif pdf_reader == 'pymupdf4llm':
        data = get_pdf_text_pymupdf4llm(pdf_docs, table_reader, image_reader)

    elif pdf_reader == 'PyPDF2':
        data = get_pdf_text_pypdf2(pdf_docs, table_reader)

    elif pdf_reader == 'pymupdf':
        data = get_pdf_text_pymupdf(pdf_docs, table_reader)

    return data