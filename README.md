# Udacity - Intro to Machine Learning

View full project [here](https://github.com/missmariss31/enron/blob/master/enronproject.md)
__________________________________________________________________________

## ENRON SCANDAL

[Summary link - Wikipedia](https://en.wikipedia.org/wiki/Enron_scandal)

>The Enron scandal was a financial scandal that eventually led to the bankruptcy of the Enron Corporation, an American energy company based in Houston, Texas, and the de facto dissolution of Arthur Andersen, which was one of the five largest audit and accountancy partnerships in the world. In addition to being the largest bankruptcy reorganization in American history at that time, Enron was cited as the biggest audit failure.

>Enron was formed in 1985 by Kenneth Lay after merging Houston Natural Gas and InterNorth. Several years later, when Jeffrey Skilling was hired, he developed a staff of executives that – by the use of accounting loopholes, special purpose entities, and poor financial reporting – were able to hide billions of dollars in debt from failed deals and projects. Chief Financial Officer Andrew Fastow and other executives not only misled Enron's Board of Directors and Audit Committee on high-risk accounting practices, but also pressured Arthur Andersen to ignore the issues.

>Enron shareholders filed a 40 billion dollar lawsuit after the company's stock price, which achieved a high of US90.75 per share in mid-2000, plummeted to less than 1 dollar by the end of November 2001. The U.S. Securities and Exchange Commission (SEC) began an investigation, and rival Houston competitor Dynegy offered to purchase the company at a very low price. The deal failed, and on December 2, 2001, Enron filed for bankruptcy under Chapter 11 of the United States Bankruptcy Code. Enron's 63.4 billion dollars in assets made it the largest corporate bankruptcy in U.S. history until WorldCom's bankruptcy the next year.

>Many executives at Enron were indicted for a variety of charges and some were later sentenced to prison. Enron's auditor, Arthur Andersen, was found guilty in a United States District Court of illegally destroying documents relevant to the SEC investigation which voided its license to audit public companies, effectively closing the business. By the time the ruling was overturned at the U.S. Supreme Court, the company had lost the majority of its customers and had ceased operating. Enron employees and shareholders received limited returns in lawsuits, despite losing billions in pensions and stock prices. As a consequence of the scandal, new regulations and legislation were enacted to expand the accuracy of financial reporting for public companies. One piece of legislation, the Sarbanes–Oxley Act, increased penalties for destroying, altering, or fabricating records in federal investigations or for attempting to defraud shareholders. The act also increased the accountability of auditing firms to remain unbiased and independent of their clients.

_______________________________________________________________________________________________________________

## Final Project: Identify Fraud from Enron Email
### By:  Marissa Schmucker
### May 2018

Below is a summary of the process in creating a POI identifier.  The full analysis, feature selection, validation, and evaluation can be found [here]('https://github.com/missmariss31/enron/blob/master/enronproject.md').

<a id='top'></a>

Table of Contents
<br><br>
[Project Goal](#Goal) | 
<br>
[Dataset Questions](#Questions) | 
<br>
[Outliers](#Outliers) | 
<br>
[Transform, Select, and Scale](#TSS) | 
<br>
[Final Analysis](#Analysis) |
<br>
[Final Thoughts](#Thoughts) | 
<br>
____________________________________________________________________________

<a id='Goal'></a>

## Project Goal

The goal of this project is to use the Enron dataset to train our machine learning algorithm to detect the possiblity of fraud (identify person's of interest.)  Since we know our persons of interest (POIs) in our dataset, we will be able to use supervised learning algorithms in constructing our POI identifier.  This will be done by picking the features within our dataset that separate our POIs from our non-POIs best.  
<br>
We will start out our analysis by answering some questions about our data.  Then, we will explore our features further by visualizing any correlations/outliers.  Next, we will transform/scale our features and select those that will be most useful in our POI identifier, engineering new features and adding them to the dataset if provided to be useful for our analysis.  We will identify at least two algorithms that may be best suited for our particular set of data and test them, tuning our parameters until optimal performance is reached.  In our final analysis, the algorithm we have fit will be validated using our training/testing data.  Using performance metrics to evaluate our analysis, any problems will be addressed and motifications made.  In our final thoughts, the performance of our final algorithm will be discussed. 
<br>

<a id='Questions'></a>

## Dataset Questions

After getting our data dictionary loaded, we can start exploring our data.  We'll answer the following questions:
<br>
1. How many people do we have in our dataset?
2. What are their names?
3. What information do we have about these people?
4. Who are the POIs in our dataset?
5. Who are the highest earners?  Are they POIs?
6. Whos stock options had the highest value (max exercised_stock_options)?
7. Are there any features we can ignore due to missing data?
8. What is the mean salary for non-POIs and POIs?
9. What features might be useful for training our algorithm?
10. Are there any features we may need to scale?

[Top](#top)

Just by looking at our dataset information, we can quickly point out a few ways to narrow down our feature selection.  Some of our features have lots of <b>missing data</b>, so those may be ones that we can remove.  Features like "restricted_stock_deferred", "loan_advances", and "director_fees" may be some that we can take out altogether.  There are also a few features that seem to be giving us the same information, like "shared_receipt_with_poi","to_messages", "from_messages", "from_this_person_to_poi", and "from_poi_to_this_person" all tell us about the person's e-mail behavior and all have the same data count, 86.  We may be able to narrow those features down to just one or two, or <b>create a new feature</b> from them (see added features in full html file.)
<br><br>
Features that may be most useful, since we're dealing with corporate fraud, are those features that tell us about the money.  Let's follow the money!  Features that will give us that money trail might be "salary", "total_payments", "exercised_stock_options", "bonus", "restricted_stock", and "total_stock_value".  We can also create new features by combigning and/or scaling our current features.

<a id='Outliers'></a>

## Outliers

When looking at the stats for poi and non-poi for the first time, I noticed that the non-poi stats were much higher than the poi stats.  That's when I remembered I didn't account for the "TOTAL" key.  So, I went back and skipped over that key when writing to my csv.  I figured I'd just pop it out of my dictionary later if I need to.  After doing that, my stats were as expected.  Our two primary outliers are persons of interest.  Since our dataset is already very small (only 18 POIs), we will keep these outliers in our dataset.

[Top](#top)

<a id='TSS'></a>

## Transform, Select, and Scale

I've selected three lists that may be useful in training our classifiers.  Each of the features selected may be able to give us some insight into the compensation and behavior of a POI.  The total compensation (total_millions) shows us that, on average, POIs are compensated more highly than non-POIs.  The same holds true for individual payments, like salary and bonus.  And, when it comes to stock behavior, POIs are more active in their exercising of stock options(exercised_stock_options.)  Other features, like from_messages, show a kind of pattern in e-mail behavior.  POIs do not send many messages.  However, the ones they do send are often to other POIs(fraction_to_poi).  These are all features we'll test before making our final feature selection.

[Top](#top)

<a id="Analysis"></a>

### FINAL FEATURES and ALGORITHM SELECTION

Our final features, based on our feature analysis and testing, will be:

- Bonus
- Exercised Stock Options
- Fraction to POI

I chose the KNeighbors Classifier since supervised learning is best for our dataset, we have a small dataset, and our POI behavior is what we're looking for(similar behavior to 'neighbors'.)  KNearest Neighbors will help us zero in on pockets of POIs/non-POIs within our testing data.  The features I selected work well with this particular classifier because 'bonus' and 'exercised stock options' are good for training the algorithm to pick up on POI compensation trends, and 'fraction_to_poi' will help our algorithm pick up on POI e-mail behavior.  I narrowed it down to three features so as not to create noise.

<a id='Thoughts'></a>

## Final Thoughts

With an accuracy of 85-95%, precision of 65-75%, and recall of 35-50%, I think our algorithm has done well considering the small amount of data used to train the algorithm.  We only had 18 POIs in our dataset and had to split our data when validating so as not to overfit our classifier.  So, that only gave us 70% of our data to train our algorithm with.  We've done pretty well, but I don't think I'd want to bet anyones life in prison on this algorithm's performance.  We could use more data!  Our Naive Bayes and Decision Tree Classifiers didn't perform as well.  Naive Bayes may have been too simple.  The Decision Tree, given the right parameters (min_samples_split=2), may have worked out.  However, results were not as impressive as our Kneighbors.  Although, overfitting is a problem as you increase the parameter number of n_neighbors in our KNeighbors Classifier.  To avoid this, I kept the default parameter setting of 5.  That left us with a working algorithm and some pretty solid evaluation metrics.  According to our confusion matrix, we were able to identify 97% of non-POIs and 50% of POIs.  I'd rather get a few POIs wrong than falsely identify a non-POI as a POI.  There may come a day when people will be convicted based on machine learning, so it's important that we be as accurate as possible.
