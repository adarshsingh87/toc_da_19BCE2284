Rashi.ipynb is jupyter-notebook file, which can be opened by installing anaocnda and jupyter-notebook.
Majority of the executed code is given, but it is not optimised for NER training.
Students need to train  NER data accurately and generate the code such that terms should be identified precisely as instructed in the toc assignment manual.

All the data operations must be performed on dataset provided.
Blacklistedfile.csv contains all pre-defined sensitivity terms.

two variants of code has been provided. 1-rashi.ipynb and 2-tocda.py& training_ner.py.  Students can choose either one.

1. student has to match all the dataset against blacklisted file and make three clusters of high_sensitive, avg_sensitive and less_sensitive.
2. After identifying such files, files should be clustered and visualized(code is given)
3. Next,students need to train the dataset using SVM(support vector machine)which will result in cyber-security related and non-cyber security related.
4.Consider all sensitive files obtained from step-1,step-2 and step-3, and then perform pos tagging on the resultant data, for which the source code is given.
5. Apart from that, student needs to extract relations as given in TOC_Assignment.doc file(refer relation builder section)
6. Custom NER labelling code is given, yet not accurate. Student needs to train custom ner model such that it identifies all the entities accurately. two types of ner code is provided and student can choose either one with the intention of improving accuracy.
7. Final step is to construct DAG from the obtained relations 



Refer following links for custom ner labelling and training

https://towardsdatascience.com/custom-named-entity-recognition-using-spacy-7140ebbb3718

https://www.machinelearningplus.com/nlp/training-custom-ner-model-in-spacy/

https://medium.com/swlh/build-a-custom-named-entity-recognition-model-ussing-spacy-950bd4c6449f

https://thinkinfi.com/prepare-training-data-and-train-custom-ner-using-spacy-python/

https://www.kaggle.com/amarsharma768/custom-ner-using-spacy

https://confusedcoders.com/data-science/deep-learning/how-to-create-custom-ner-in-spacy




svm training links:

https://medium.com/@bedigunjit/simple-guide-to-text-classification-nlp-using-svm-and-naive-bayes-with-python-421db3a72d34

https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a

https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/ },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
ab7b4aa090d078a6a4bf8774-0-20\" stroke-width=\"2px\" d=\"M1470,352.0 C1470,2.0 3725.0,2.0 3725.0,352.0\" fill=\"none\" stroke=\"currentColor\"/>\n",
       "    <text dy=\"1.25em\" style=\"font-size: 0.8em; letter-spacing: 1px\">\n",
       "        <textPath xlink:href=\"#arrow-0f591c5fab7b4aa090d078a6a4bf8774-0-20\" class=\"displacy-label\" startOffset=\"50%\" side=\"left\" fill=\"currentColor\" text-anchor=\"middle\">conj</textPath>\n",
       "    </text>\n",
       "    <path class=\"displacy-arrowhead\" d=\"M3725.0,354.0 L3733.0,342.0 3717.0,342.0\" fill=\"currentColor\"/>\n",
       "</g>\n",
       "\n",
       "<g class=\"displacy-arrow\">\n",
       "    <path class=\"displacy-arc\" id=\"arrow-0f591c5fab7b4aa090d078a6a4bf8774-0-21\" stroke-width=\"2px\" d=\"M3745,352.0 C3745,264.5 3885.0,264.5 3885.0,352.0\" fill=\"none\" stroke=\"currentColor\"/>\n",
       "    <text dy=\"1.25em\" style=\"font-size: 0.8em; letter-spacing: 1px\">\n",
       "        <textPath xlink:href=\"#arrow-0f591c5fab7b4aa090d078a6a4bf8774-0-21\" class=\"displacy-label\" startOffset=\"50%\" side=\"left\" fill=\"currentColor\" text-anchor=\"middle\">cc</textPath>\n",
       "    </text>\n",
       "    <p