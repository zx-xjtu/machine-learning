
This documentation is prepared as the workflow to accompany the following study:

"Machine Learning for Screening Small Molecules as Passivation Materials for Enhanced Perovskite Solar Cells"
If you have questions or suggestions, please contact Bo Chen at bochen@xjtu.edu.cn and Xin Zhang zx13460208639@163.com

1 Installation

To run these codes, please install softwares including Anaconda and PyCharm

Besides, we need to install some packages in PyCharm platform:

install.packages("Numpy")

install.packages("Pandas")

install.packages("seaborn")

install.packages("matplotlib")

install.packages("scikit-learn")



2 Datasets
The dataset can be obtained from 

3 Machine-learning classification model
support vector machine
neural network model 
random forest
k-nearest neighbor
naive Bayes
We evaluate the models using a novel Random-Extracted and Recoverable Cross-Validation method (RE-RCV) proposed by us.

4 Model prediction

After the models are trained, we can load the dataset that consists of traits for the molecules we want to predict. Then, we can predict the passivation effect of the molecule.
