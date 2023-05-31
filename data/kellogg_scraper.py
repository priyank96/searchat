import requests
import re
import pandas as pd

from tqdm import tqdm
from bs4 import BeautifulSoup

courses = []
descriptions = []

homepage_url = "https://www6.kellogg.northwestern.edu/CourseCatalog/coursecatalog/catalogsearchscreen"
description_url="https://www6.kellogg.northwestern.edu/coursecatalog/coursecatalog/coursedescription?CourseCatalogID="

response = requests.get(homepage_url)
soup = BeautifulSoup(response.text, 'html.parser')
pattern = re.compile(r'^course-.*$')
course_links = soup.find_all('a', id=pattern)

for link in tqdm(course_links):
  # get course name
  sibling_div = link.find_next_sibling('div')
  courses.append(sibling_div.text.strip())
  # get course description
  response = requests.post(description_url+link['id'][7:])
  descriptions.append(response.text)

pd.DataFrame({'Course Name': courses,'Course Description': descriptions}).to_csv('courses.csv',index=False)