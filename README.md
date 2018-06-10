# Applied-Machine-Learning



Project 1 - Recommendation Systems
----------------------------------
In this project we will implement and evaluate algorithms to learn a matrix factorization model from explicit ratings data.


Project 2 - Word Embeddings
---------------------------
In this project we will implement a word2vec word embedding model.


QUESTIONS FOR GAL!!!!!!!

1. in algorithm: Sample a word context pair (wt; wc) from dataset
   --- should we choose input(target) word from 1...N and than the context is only relevant to the index we chose, or is the context relevant to ALL the indices that the target_word appears in the corpus N ?
   
איך בוחרים W_T ? לבחירתינו:
* מגרילים שורה (יחסית לאורך המשפטים), ואז מגרילים מילה
* פורסים את כל המילים בשורה אחת ומגרילים  - זה מספיק טוב   
   
2. in algorithm: for i = 1; : : : ; MINIBATCHSIZE do
   --- do we do this loop MINIBATCHSIZE times? why??
   
3. in algorithm: 4. implement a function that uses the above function to create a mini-batch of context/input pairs
    --- does this means that if C=3 than mini-batch = one or two pairs of (target,context) ? and what if C=1? than mini-batch=1?

* minibatch is about the SGD - after how mush iterations do we make update? after MINIBATCHSIZE * 2C !!!





TO FIX: 
* TODO: DAN : alg.LearnParamsUsingSGD        
NEED TO insert the REDUCING OF alpha !!!!
      After a fixed number of iterations (specified in the hyperparameters) has elapsed the learning rate should be reduced by 50%
      
      
* TODO: continue and fix structure of model.hyper or alg.hyper ---> beacause in section 7.2 he asks for different parameters...


Hyperparameters can sometimes have a great effect on generalization performance. We will
consider the effect of the embedding size and other hyperparams.
Deliverable 2. Set the hyper params as follows:
1. Learning Rate = 0.3
2. Num iterations = 20000
3. Maximum context window size = 5
4. mini-batch size = 50
5. noise distribution = Unigram (α = 1)
6. number of negative samples per context/input pair (denoted K above)=10
Vary the size of the word embedding ,d from 10 to 300 in 5 evenly spaced intervals. In two
separate plots, plot both training time and train and test (mean) log-likelihood as a function of
d. All hyper-parameter configurations of the algorithm should be clearly specified.
