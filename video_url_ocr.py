import cv2
import os
from os import listdir
from os.path import isfile, join
import pytesseract
import ntpath
import pandas as pd
import numpy as np
import re
from natsort import natsorted

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"

####### VIDEO SECTION ###############
####### changes videos in a path into folders of images######
def video_to_image(vid,num):
    '''
    Parses a Video into second-based images base on FPS rate and writes them into a folder with the same name
    :param vid: an flv video file
    :param num: number of frames per second
    :return: None
    '''

    # create a folder with the same name as video
    my_image_path = 'C:\\video_images'
    path, file_name = os.path.split(vid)
    folder = join(my_image_path , file_name)
    os.makedirs(folder)
    # capture the video
    vidcap = cv2.VideoCapture(vid)

    # loop to read all frames and write one per second to the folder created above
    success, image = vidcap.read()
    count = 0
    while success:
        if count % num == 0:
            frame = file_name[:-3] + str(count) + '.jpg'

            cv2.imwrite(join(folder, frame), image)  # save frame as JPEG file to the folder
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        count += 1

# example of reading a group of videos in a folder and converting them into folders of images per seconds
mypath = "C:\\videos"
onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
for file in onlyfiles:
    video_to_image(file,30)




########### IMAGE-BASE SECTION ##################



###### crop images base on the browser ancher to get url area in each folder #######
path_to_anchor1 = 'C:\\Users\\mh4pk\\Downloads\\temp.jpg'
path_to_anchor2 = 'C:\\Users\\mh4pk\\Downloads\\onion_vid.jpg'
def AOI(img, path_to_anchor1, path_to_anchor2):
    '''
    Uses  openCV TM_CCOEFF_NORMED algorithm to match anchors for text area on top of browser
    (example is onion symbol on Tor browser) this function is designed for two anchors.
     The first one for chrome and the second one for Tor. If the result of matching pass the threshold of 0.75 or 0.8
    :param img: original image to search with template matching for anchors (image matrix)
    :param path_to_anchor1: path to chrome anchor (string)
    :param path_to_anchor2: path to tor anchor  (string)
    :return: Tuple (Aoi_window_G (image matrix), Aoi_window_T (image matrix), is_chrome(boolean), is_tor(boolean))
    '''
    # img = cv2.imread(complete_filename,1)
    chrome_match_pr = []
    tor_match_pr = []
    template_G = cv2.imread(path_to_anchor1, 1)
    template_shape_G = template_G.shape[0:2]
    w_G, h_G = template_shape_G[::-1]

    template_T = cv2.imread(path_to_anchor2, 1)
    template_shape_T = template_T.shape[0:2]
    w_T, h_T = template_shape_T[::-1]
    method = eval('cv2.TM_CCOEFF_NORMED')
    is_tor = True
    is_chrome = True
    #template matching for chrome anchor
    res_G = cv2.matchTemplate(img, template_G, method)
    min_val, max_val_G, min_loc, max_loc_G = cv2.minMaxLoc(res_G)
    hight = h_G
    AOI_window_G = img[max_loc_G[1] - 2:max_loc_G[1] + hight, :max_loc_G[0]]
    print("the value of chrome max is : " + str(max_val_G))
    chrome_match_pr.append(max_val_G)
    # check threshold for chrome  with o.75
    if max_val_G < 0.65:
        max_loc_G = (2, 0)
        hight = 2
        AOI_window_G = img[max_loc_G[0] - 1:max_loc_G[0] + hight, 0:50]
        is_chrome = False
    # template matching for Tor anchor
    res_T = cv2.matchTemplate(img, template_T, method)
    min_val, max_val_T, min_loc, max_loc_T = cv2.minMaxLoc(res_T)
    hight = h_T
    AOI_window_T = img[max_loc_T[1] - 2:max_loc_T[1] + hight, max_loc_T[0] + w_T:]
    print("the value of tor max is : " + str(max_val_T))
    tor_match_pr.append(max_val_T)
    # check threshold for chrome  with o.75
    if max_val_T < 0.75:
        max_loc_T = (2, 0)
        hight = 2
        AOI_window_T = img[max_loc_T[0] - 1:max_loc_T[0] + hight, 0:50]
        is_tor = False
    return AOI_window_G, AOI_window_T, is_chrome, is_tor
    # can write any of these image cropped in a file instead of returning ex: cv2.imwrite('C:\\Users\\Moji\\Desktop\\last2.png', AOI_window)

def sharpen(img):
    '''
    Gets a gray-scale image and sharpens it using one of the two kernels
        :parm img: an image matrix gray-scale than should be sharpened
        :return sharpened : sharpened image matrix gray-scale
    '''
    kernel = np.array([[-0.00391, -0.01563, -0.02344, -0.01563, -0.00391],
                       [-0.01563, -0.06250, -0.09375, -0.06250, -0.01563],
                       [-0.02344, -0.09375, 1.85980, -0.09375, -0.02344],
                       [-0.01563, -0.06250, -0.09375, -0.06250, -0.01563],
                       [-0.0391, -0.01563, -0.02344, -0.01563, -0.00391]])
    '''
    kernel =np.zeros((9,9),np.float)
    kernel[4,4] =2.0
    boxFilter = np.ones((9,9),np.float)/81.0
    kernel = kernel-boxFilter
    '''
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened


def url_area_to_text(url_pics_directory):
    '''
    This function uses multiple image processing techniques and tesseract OCR to grab the url text out of an image
    :param url_pics_directory: a path to a folder of url cropped images (string)
    :return: texts: list of all texts  after processing all url images in a directory using tesseract OCR
    '''
    texts = []
    counter = 0
    url_images = url_pics_directory
    for filename in listdir(url_images):
        img = cv2.imread(join(url_images, filename))
        gray_scaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_scaled2 = cv2.resize(gray_scaled, (img.shape[1] * 3, img.shape[0] * 3))
        denoised_gray = cv2.fastNlMeansDenoising(gray_scaled2, None, 10, 7, 21)
        sharpened_img = sharpen(denoised_gray)
        d_kernel = np.ones((3, 3), np.uint8)
        dialation = cv2.dilate(sharpened_img, d_kernel, iterations=1)
        text = pytesseract.image_to_string(sharpened_img, lang='eng')
        counter += 1
        print(text)
        texts.append(text)
    return texts
def url_cleaner_1(url_texts):
    '''
    Replaces wrong results of www with www and adds a dot before com when OCR missed it.
    :param url_texts: unclean text (string)
    :return: cleaner_urls: url text cleaner(string)
    '''
    cleaner_urls = []
    wrong_domains = ['wvw', 'wwv', 'vww']
    for text in url_texts:

        for domain in wrong_domains:
            if text.find(domain):
                text = text.replace(domain, 'www')
        if text.find('www') != -1 and (text.find('www') + 3) < len(text) and text[text.find('www') + 3] != '.':
            Position = text.find('www') + 3

            text = text[:Position] + '.' + text[Position:]
        if text.find('com') != -1:
            Position = text.find('com') - 1

            text = text[:Position] + '.' + text[Position + 1:]

        cleaner_urls.append(text)
    return cleaner_urls


def url_cleaner_2(url_texts):
    '''
    Replaces other possible mistakes by OCR with correct characters
    :param url_texts: unclean text (string)
    :return: cleaner_urls: url text cleaner(string)
    '''
    cleaned_urls = []
    for text in url_texts:
        if text.find('http://') != -1 or text.find('https://') != -1:
            begin_url = text[text.rfind('http'):]
        elif text.find('www') != -1:
            begin_url = text[text.rfind('www'):]
        elif text.find('://') != -1:
            begin_url = text[text.rfind('://') + 3:]
        else:
            begin_url = text
        if begin_url.find('\n') != -1:
            end = begin_url.find('\n')
        else:
            end = len(begin_url) - 1
        cleaned_urls.append(begin_url[:end])
    return cleaned_urls


def url_finder_regex(urls):
    '''
    Uses Regex to select the correct url and domain from the text
    :param urls: cleaner url as a string
    :return: Tuple (regex_urls: grabbed url from text (string), regex_domains: grabbed domain from text (string))
    '''
    regex_urls = ['No URL'] * len(urls)
    regex_domains = [''] * len(urls)
    url_counter = 0
    url_pattern = re.compile(
        r'((http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?([a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?)(\/.*)?)(\n)?.*$')
    for url in urls:

        URL_match = url_pattern.finditer(url)
        for match in URL_match:
            regex_urls[url_counter] = match.group(0)
            domain = match.group(3)
            pos = 0
            if domain.find('www.') != -1:
                pos = domain.find('www.') + 4
            regex_domains[url_counter] = domain[pos:]
        url_counter += 1
    return regex_urls, regex_domains



def screenshots_to_csv(screenshots_path):
    '''
    Gets a folder path to all the screen shots and creats two folder one for using chrome and the other for using chrome.
    It fills those with either the detected text area filled or empty image also runs tesseract function.
     do post processing and adds the finall result to a csv file
    :param screenshots_path: path of screenshot folder
    :return: None
    '''
    general_path, directory = os.path.split(screenshots_path)
    crop_path = 'C:\\cropped\\' + directory + '_cropped'

    onlyfiles = [join(screenshots_path, f) for f in listdir(screenshots_path) if isfile(join(screenshots_path, f))]
    # creates two folders for both chrome and tor cropped text fields
    file_names = []
    os.makedirs(join(crop_path, 'cropped_c'))
    os.makedirs(join(crop_path, 'cropped_t'))
    cropped_chrome_dir = join(crop_path, 'cropped_c')
    cropped_tor_dir = join(crop_path, 'cropped_t')
    file_number = 1
    tor_used = []
    chrome_used = []

    # crop images and fill the two folders with results of AOI function
    #  also fills two lists of booleans for chrome or tor usage
    for file in onlyfiles:
        img = cv2.imread(file)
        h, w, c = img.shape
        crop_img = img[0:int(h / 3), 0:w]
        chrome_area, tor_area, is_chrome, is_tor = AOI(crop_img, path_to_anchor1, path_to_anchor2)
        head, tail = ntpath.split(file)
        file_names.append(tail)
        cv2.imwrite(join(cropped_chrome_dir, tail), chrome_area)

        cv2.imwrite(join(cropped_tor_dir, tail), tor_area)
        chrome_used.append(is_chrome)
        tor_used.append(is_tor)
        file_number += 1
    user = [directory[:-4]] * len(tor_used)
    # uses tesseract to create text.
    chrome_texts = url_area_to_text(cropped_chrome_dir)
    tor_texts = url_area_to_text(cropped_tor_dir)
    # post processing on the text
    cleaner_tor_urls = url_cleaner_1(tor_texts)
    cleaner_tor_urls = url_cleaner_2(cleaner_tor_urls)
    tor_urls, tor_domains = url_finder_regex(cleaner_tor_urls)
    cleaner_chrome_urls = url_cleaner_1(chrome_texts)
    cleaner_chrome_urls = url_cleaner_2(cleaner_chrome_urls)
    chrome_urls, chrome_domains = url_finder_regex(cleaner_chrome_urls)

    # write dataframe to a cvs file
    user_url_dataframe = pd.DataFrame(
        {"user_ID": user, "Image": file_names, "tor used": tor_used, "chrome used": chrome_used,
         "Tor URL": tor_urls, "Chrome URL": chrome_urls, "Tor Url uncleaned": tor_texts,
         "Chrome Url uncleaned": chrome_texts, "Tor Domain": tor_domains, "Chrome Domain": chrome_domains})
    data_name = directory[:-4] + '_data.csv'
    user_url_dataframe.to_csv(join(crop_path, data_name))


##########
    '''
# example of running the whole process on a folder of screenshots folders
image_folder_path = 'C:\\video_images'
for file in listdir(image_folder_path):
    screenshots = join(image_folder_path, file)
    screenshots_cropper(screenshots)

    '''
###########
screenshots_to_csv('C:\\video_images\\AEA014_2018-04-16 09-19-22.flv')
