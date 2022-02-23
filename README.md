# How to use AI to help you get a job ? [Medium](https://medium.com/p/e6a2894914e5)

### Introduction
Using AI model to cauculate match score between a resume and job description. 
The AI model will extract keywords and find important paragraph from a job description, and you can use those information to modify you resume, highlight the skills which is relatived to the job you want to apply.  

---

### Model information 
- [Gensim](https://radimrehurek.com/gensim/) - an open-source library for unsupervised topic modeling and natural language processing, using modern statistical machine learning.

---

### How to  - Example 

- **Match Score**

  ```python
  inputText = [resumeText, jobDesciptionText]
  cv = CountVectorizer()
  count_matrix = cv.fit_transform(inputText)
  matchPercentage = round(cosine_similarity(count_matrix)[0][1] * 100, 2)
  print(f"[ Resume Match Score ]\n{matchPercentage}")
  ```
  output
  ```
  MatchScore: 58.98 😅
  ```

- **Keyword Extraction**

  ```python
  from gensim.summarization import keywords
  keywords(despText, ratio=0.1)
  ```
  output
  ```
  [ Keywords ] 
  data, business, experience, experiences, models, modeling, model, statistical, statistics, development, develop, techniques, regression, tools, insights, analysis, tree, trees, outcomes, job, company
  ```

- **Text Summarization**
  
  ```python
  from gensim.summarization.summarizer import summarize
  outputSummary = summarize(jobDesciptionText, ratio=0.1)
  ```
  output
  ```
  [ Extractive Text Summarization ]
  We are looking for a Data Scientist who will support our product, sales, leadership and marketing teams with insights gained from analyzing company data. The ideal candidate is adept at using large data sets to find opportunities for product and process optimization and using models to test the effectiveness of different courses of action. Mine and analyze data from company databases to drive optimization and improvement of product development, marketing techniques and business strategies.
  ```
