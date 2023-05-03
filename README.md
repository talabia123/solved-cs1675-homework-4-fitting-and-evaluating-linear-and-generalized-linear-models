Download Link: https://assignmentchef.com/product/solved-cs1675-homework-4-fitting-and-evaluating-linear-and-generalized-linear-models
<br>
This assignment is focused on fitting and evaluating linear and generalized linear models. You will work with simple linear relationships, as well as basis functions. You will evaluate model performance on training and test sets, and see how those comparisons relate to assessing model performance using the Evidence (marginal likelihood).

Completing this assignment requires filling in missing pieces of information from existing code chunks, programming complete code chunks from scratch, typing discussions about results, and working with LaTeX style math formulas. A template .Rmd file is available to use as a starting point for this homework assignment. The template is available on CourseWeb.

<strong>IMPORTANT:</strong> Please pay attention to the eval flag within the code chunk options. Code chunks with eval=FALSE will <strong>not</strong> be evaluated (executed) when you Knit the document. You <strong>must</strong> change the eval flag to be eval=TRUE. This was done so that you can Knit (and thus render) the document as you work on the assignment, without worrying about errors crashing the code in questions you have not started. Code chunks which require you to enter all of the required code do not set the eval flag. Thus, those specific code chunks use the default option of eval=TRUE.

<h2>Load packages</h2>

This assignment uses the dplyr and ggplot2 packages, which are loaded in the code chunk below. The assignment also uses the tibble package to create tibbles, and the readr package to load CSV files, and the purrr package for functional programming. All of the listed packages are part of the tidyverse and so if you downloaded and installed the tidyverse already, you will have all of these packages. This assignment will use the MASS package to generate random samples from a MVN distribution. The MASS package should be installed with base R, and is listed with the System Library set of packages.

library(dplyr)library(ggplot2)

<h2>Problem 01</h2>

This problem is focused on setting up the log-posterior function for a linear model. You will program the function using matrix math, such that you can easily scale your code from a linear relationship with a single input up to complex linear basis function models. You will assume independent Gaussian priors on all (boldsymbol{beta})-parameters with a shared prior mean (mu_{beta}) and shared prior standard deviation, (tau_{beta}). An Exponential prior with rate parameter (lambda) will be assumed for the likelihood noise, (sigma). The complete probability model for the response, (y_n), is shown below using the linear basis notation. The (n)-th row of the basis design matrix, (boldsymbol{Phi}) is denoted as (boldsymbol{phi}_{n,:}). It is assumed that the basis is of order (J).

[ y_n mid mu_n, sigma sim mathrm{normal}left(y_n mid mu_n, sigmaright) ]

[ mu_n = boldsymbol{phi}_{n,:}boldsymbol{beta} ]

[ boldsymbol{beta} mid mu_{beta}, tau_{beta} sim prod_{j=0}^{J}left( mathrm{normal}left( beta_j mid mu_{beta}, tau_{beta} right) right) ]

[ sigma mid lambda sim mathrm{Exp}left(sigma mid lambda right) ]

<h3>1a)</h3>

The code chunk below reads in a data set consisting of two variables, an input x and a response y. As shown by the glimpse() of the data set, there are 50 observations of the two continuous variables.

train_01 &lt;- readr::read_csv(“https://raw.githubusercontent.com/jyurko/CS_1675_Spring_2020/master/hw_data/hw04/train_01.csv”, col_names = TRUE)## Parsed with column specification:## cols(##   x = col_double(),##   y = col_double()## )train_01 %&gt;% glimpse()## Observations: 50## Variables: 2## $ x &lt;dbl&gt; -1.10210335, -0.71797339, 1.19319957, 1.55483625, 0.98823320…## $ y &lt;dbl&gt; 1.66107092, 0.57348497, -1.17436302, -1.33294329, -0.5224802…

<h4>PROBLEM</h4>

<strong>Create a scatter plot between the response and the input using </strong><strong>ggplot()</strong><strong>. In addition to using </strong><strong>geom_point()</strong><strong>, include a </strong><strong>geom_smooth()</strong><strong> layer to your graph. Set the </strong><strong>method</strong><strong> argument to </strong><strong>‘lm’</strong><strong> in the call to </strong><strong>geom_smooth()</strong><strong>. Based on the figure what type of relationship do you think exists between the response and the input?</strong>

<h4>SOLUTION</h4>

?

### your code here

<h3>1b)</h3>

In your response to Problem 1a), you should see a “best fit line” and it’s associated confidence interval displayed with the scatter plot. Behind the scenes, ggplot2() fits a linear model between the response and the input with Maximum Likelihood Estimation, and plots the result on the figure. You will now work through a full Bayesian linear model. Before coding the log-posterior function, you will start out by creating the list of required information, info_01, which defines the data and hyperparameters that you will ultimately pass into the log-posterior function.

You will need to create a design matrix assuming a linear relationship between the input and the response. The mean trend function is written for you below:

[ mu_n = beta_0 + beta_1 x_{n} ]

<h4>PROBLEM</h4>

<strong>Create the design matrix assuming a linear relationship between the input and the response, and assign the object to the </strong><strong>Xmat_01</strong><strong> variable. Complete the </strong><strong>info_01</strong><strong> list by assigning the response to </strong><strong>yobs</strong><strong> and the design matrix to </strong><strong>design_matrix</strong><strong>. Specify the shared prior mean, </strong><strong>mu_beta</strong><strong>, to be 0, the shared prior standard deviation, </strong><strong>tau_beta</strong><strong>, as 5, and the rate parameter on the noise, </strong><strong>sigma_rate</strong><strong>, to be 1.</strong>

<h4>SOLUTION</h4>

Xmat_01 &lt;-  info_01 &lt;- list(  yobs = ,  design_matrix = ,  mu_beta = ,  tau_beta = ,  sigma_rate = )

<h3>1c)</h3>

You will now define the log-posterior function lm_logpost(). You will continue to use the log-transformation on (sigma), and so you will actually define the log-posterior in terms of the mean trend (boldsymbol{beta})-parameters and the unbounded noise parameter, (varphi = logleft[sigmaright]).

The comments in the code chunk below tell you what you need to fill in. The unknown parameters to learn are contained within the first input argument, unknowns. You will assume that the unknown (boldsymbol{beta})-parameters are listed before the unknown (varphi) parameter in the unknowns vector. You will assume that all variables contained in the my_info list (the second argument to lm_logpost()) are the same fields in the info_01 list you defined in Problem 1b).

<h4>PROBLEM</h4>

<strong>Define the log-posterior function by completing the code chunk below. You must calculate the mean trend, </strong><strong>mu</strong><strong>, using matrix math between the design matrix and the unknown </strong><strong>(boldsymbol{beta})</strong><strong> column vector. After you complete the function, test that it out by evaluating the log-posterior at two different sets of parameter values. Try out values of -1 for all parameters, and then try out values of 1 for all parameters.</strong>

<em>HINT</em>: If you have successfully completed the log-posterior function, you should get a value of -296.5826 for the -1 guess values, and a value of -123.9015 for the +1 guess values.

<em>HINT</em>: Don’t forget about useful data type conversion functions such as as.matrix() and as.vector().

<h4>SOLUTION</h4>

lm_logpost &lt;- function(unknowns, my_info){  # specify the number of unknown beta parameters  length_beta &lt;-     # extract the beta parameters from the `unknowns` vector  beta_v &lt;-     # extract the unbounded noise parameter, varphi  lik_varphi &lt;-     # back-transform from varphi to sigma  lik_sigma &lt;-     # extract design matrix  X &lt;-     # calculate the linear predictor  mu &lt;-     # evaluate the log-likelihood  log_lik &lt;-     # evaluate the log-prior  log_prior_beta &lt;-     log_prior_sigma &lt;-     # add the mean trend prior and noise prior together  log_prior &lt;-     # account for the transformation  log_derive_adjust &lt;-     # sum together  }lm_logpost( ) lm_logpost( )

<h3>1d)</h3>

The my_laplace() function is started for you in the code chunk below. You will complete the missing parts and then you will fit the Bayesian linear model from a startig guess of zero for all parameters.

<h4>PROBLEM</h4>

<strong>Complete the </strong><strong>my_laplace()</strong><strong> function below and then fit the Bayesian linear model using a starting guess of zero for all parameters. Print the posterior mode and posterior standard deviations to the screen. Should you be concerned about the initial guess impacted the posterior results?</strong>

<h4>SOLUTION</h4>

my_laplace &lt;- function(start_guess, logpost_func, …){  # code adapted from the `LearnBayes“ function `laplace()`  fit &lt;- optim(start_guess,               logpost_func,               gr = NULL,               …,               method = “BFGS”,               hessian = TRUE,               control = list(fnscale = -1, maxit = 1001))    mode &lt;-   h &lt;-   p &lt;- length(mode)  int &lt;-     list(mode = mode,       var_matrix = h,       log_evidence = int,       converge = ifelse(fit$convergence == 0,                         “YES”,                          “NO”),       iter_counts = fit$counts[1])}

Fit the Bayesian linear model.

laplace_01 &lt;- my_laplace( )

Display the posterior modes and posterior standard deviations.

### your code here

?

<h3>1e)</h3>

The generate_lm_post_samples() function is started for you in the code chunk below. The first argument, mvn_result, is the Laplace Approximation result object returned from the my_laplace() function. The second argument, length_beta, specifies the number of mean trend (boldsymbol{beta})-parameters to the model. The last argument, num_samples, specifies the total number of posterior samples to generate. The function is nearly complete, except you must back-transform from the unbounded (varphi) parameter to the noise (sigma). After completing the function, generate 2500 posterior samples from your model stored in the laplace_01 object. Be careful to specify the number of (beta)-parameters correctly!

After generating the posterior samples you will study the posterior distribution on the slope, (beta_1).

<h4>PROBLEM</h4>

<strong>Complete the </strong><strong>generate_lm_post_samples()</strong><strong> function by back-transforming from </strong><strong>varphi</strong><strong> to </strong><strong>sigma</strong><strong>. After completing the function, generate 2500 posterior samples from your </strong><strong>laplace_01</strong><strong> model. Create a histogram with 55 bins using </strong><strong>ggplot2()</strong><strong> for the slope, </strong><strong>beta_01</strong><strong>, and then calculate the probability that the slope is positive.</strong>

<h4>SOLUTION</h4>

generate_lm_post_samples &lt;- function(mvn_result, length_beta, num_samples){  MASS::mvrnorm(n = num_samples,                 mu = mvn_result$mode,                 Sigma = mvn_result$var_matrix) %&gt;%     as.data.frame() %&gt;% tbl_df() %&gt;%     purrr::set_names(c(sprintf(“beta_%02d”, (1:length_beta) – 1), “varphi”)) %&gt;%     mutate(sigma = )}

Generate posterior samples.

set.seed(87123)post_samples_01 &lt;- generate_lm_post_samples( )

Create the posterior histogram on (beta_1).

###

The posterior probability that the slope is greater than zero is:

###

<h2>Problem 02</h2>

Now that you can fit a Bayesian linear model, it’s time to work with making posterior predictions from the model. You will use those predictions to calculate and summarize the errors of the model relative to observations. Since RMSE and R-squared have been discused throughout lecture, you will work with the Mean Absolute Error (MAE) metric.

<h3>2a)</h3>

The code chunk below starts the post_lm_pred_samples() function. This function generates posterior mean trend predictions and posterior predictions of the response. The first argument, Xnew, is a test design matrix. The second argument, Bmat, is a matrix of posterior samples of the (boldsymbol{beta})-parameters, and the third arugment, sigma_vector, is a vector of posterior samples of the likelihood noise. The Xnew matrix has rows equal to the number of predictions points, M, and the Bmat matrix has rows equal to the number of posterior samples S.

You must complete the function by performing the necessary matrix math to calculate the matrix of posterior mean trend predictions, Umat, and the matrix of posterior response predictions, Ymat. You must also complete missing arguments to the definition of the Rmat and Zmat matrices. The Rmat matrix replicates the posterior likelihood noise samples the correct number of times. The Zmat matrix is the matrix of randomly generated standard normal values. You must correctly specify the required number of rows to the Rmat and Zmat matrices.

The post_lm_pred_samples() returns the Umat and Ymat matrices contained within a list.

<h4>PROBLEM</h4>

<strong>Perform the necessary matrix math to calculate the matrix of posterior predicted mean trends </strong><strong>Umat</strong><strong> and posterior predicted responses </strong><strong>Ymat</strong><strong>. Specify the number of required rows to create the </strong><strong>Rmat</strong><strong> and </strong><strong>Zmat</strong><strong> matrices.</strong>

<h4>SOLUTION</h4>

post_lm_pred_samples &lt;- function(Xnew, Bmat, sigma_vector){  # number of new prediction locations  M &lt;- nrow(Xnew)  # number of posterior samples  S &lt;- nrow(Bmat)    # matrix of linear predictors  Umat &lt;-     # assmeble matrix of sigma samples  Rmat &lt;- matrix(rep(sigma_vector, M), , byrow = TRUE)    # generate standard normal and assemble into matrix  Zmat &lt;- matrix(rnorm(M*S), , byrow = TRUE)    # calculate the random observation predictions  Ymat &lt;-     # package together  list(Umat = Umat, Ymat = Ymat)}

<h3>2b)</h3>

The code chunk below is completed for you. The function make_post_lm_pred() is a wrapper which calls the post_lm_pred_samples() function. It contains two arguments. The first, Xnew, is a test design matrix. The second, post, is a data.frame of posterior samples. The function extracts the (boldsymbol{beta})-parameter posterior samples and converts the object to a matrix. It also extracts the posterior samples on (sigma) and converts to a vector.

make_post_lm_pred &lt;- function(Xnew, post){  Bmat &lt;- post %&gt;% select(starts_with(“beta_”)) %&gt;% as.matrix()    sigma_vector &lt;- post %&gt;% pull(sigma)    post_lm_pred_samples(Xnew, Bmat, sigma_vector)}

You now have enough pieces in place to generate posterior predictions from your model.

<h4>PROBLEM</h4>

<strong>Make posterior predictions on the training set. What are the dimensions of the returned </strong><strong>Umat</strong><strong> and </strong><strong>Ymat</strong><strong> matrices? Do the columns correspond to the number of prediction points?</strong>

<em>HINT</em>: The make_post_lm_pred() function returns a list. To access the variables or fields of a list use the $ operator.

<h4>SOLUTION</h4>

Make posterior predictions on the training set.

post_pred_samples_01 &lt;-

The dimensionality of the posterior predicted mean trend matrix is:

###

The dimensionality of the posterior predicted response matrix is:

###

<h3>2c)</h3>

You will now use the model predictions to calculate the error between the model and the training set observations. Since you generated 2500 posterior samples, you have 2500 different sets of predictions! So, to get started you will focus on the first 3 posterior samples.

<h4>PROBLEM</h4>

<strong>Calculate the error between the predicted mean trend and the training set observations for each of the first 3 posterior predicted samples. Assign the errors to separate vectors, as indicated in the code chunk below.</strong>

<strong>Why are you considering the mean trend when calculating the error with the response, and not the predicted response values?</strong>

<h4>SOLUTION</h4>

The error between the first 3 posterior predicted mean trend samples and the training set observations are calculated below.

### error of the first posterior sampleerror_01_post_01 &lt;-  ### error of the second posterior sampleerror_01_post_02 &lt;-  ### error of the third posterior sampleerror_01_post_03 &lt;-

?

<h3>2d)</h3>

You will now calculate the Mean Absolute Error (MAE) associated with each of the three error samples calculated in Problem 2c). However, before calculating the MAE, first consider the dimensionality of the error_01_post_01. What is the length of the error_01_post_01 vector? When you take the absolute value and then average across all elements in that vector, what are you averaging over?

<h4>PROBLEM</h4>

<strong>What is the length of the </strong><strong>error_01_post_01</strong><strong> vector? Calculate the MAE associated with each of the 3 error vectors you calculated in Problem 2c. What are you averaging over when you calculate the mean absolute error? Are the three MAE values the same? If not, why would they be different?</strong>

<em>HINT</em>: The absolute value can be calculated with the abs() function.

<h4>SOLUTION</h4>

?

###

Now calculate the MAE associated with each of the first three posterior samples.

mae_01_post_01 &lt;-  mae_01_post_02 &lt;-  mae_01_post_03 &lt;-

?

<h3>2e)</h3>

In Problem 2d) you calculated the MAE associated with the first 3 posterior samples. However, you can calculate the MAE associated with every posterior sample. Although it might seem like you need to use a for-loop to do so, R will simplify the operation for you. If you perform an addition or subtraction between a matrix and a vector, R will find the dimension that that matches between the two and then repeat the action over the other dimension. Consider the code below, which has a vector, a_numeric, subtracted from a matrix a_matrix:

a_matrix – a_numeric

Assuming that a_matrix has 10 rows and 25 columns and a_numeric is length 10, R will subtract a_numeric from each column in a_matrix. The result will be another matrix with the same dimensionality as a_matrix. To confirm this is the case, consider the example below where a vector of length 2 is subtracted from a matrix of 2 rows and 4 columns. The resulting dimensionality is 2 rows by 4 columns.

### a 2 x 4 matrixmatrix(1:8, nrow = 2, byrow = TRUE)##      [,1] [,2] [,3] [,4]## [1,]    1    2    3    4## [2,]    5    6    7    8### a vector length 2c(1, 2)## [1] 1 2### subtracting the two yields a matrixmatrix(1:8, nrow = 2, byrow = TRUE) – c(1, 2)##      [,1] [,2] [,3] [,4]## [1,]    0    1    2    3## [2,]    3    4    5    6

You will use this fact to calculate the error associated with each training point and each posterior sample all at once.

<h4>PROBLEM</h4>

<strong>Calculate the absolute value of the error between the mean trend matrix and the training set response. Print the dimensionality of the </strong><strong>absE01mat</strong><strong> matrix to screen.</strong>

<h4>SOLUTION</h4>

?

absE01mat &lt;-  ### dimensions?

<h3>2f)</h3>

You must now summarize the absolute value errors by averaging them appropriately. Should you average across the rows or down the columns? In R the colMeans() will calculate the average value associated with each column in a matrix and returns a vector. Likewise, the rowMeans() function calculates the average value along each row and returns a vector. Which function should you use to calculate the MAE associated with each posterior sample?

<h4>PROBLEM</h4>

<strong>Calculate the MAE associated with each posterior sample and assign the result to the </strong><strong>MAE_01</strong><strong> object. Print the data type (the class) of the </strong><strong>MAE_01</strong><strong> to the screen and display its length. Check your result is consistent with the MAEs you previously calculated in Problem 2d).</strong>

<h4>SOLUTION</h4>

?

MAE_01 &lt;-  class() ### length?

Check with the results you calculated previously.

### your code here

<h3>2g)</h3>

You have calculated the MAE associated with each posterior sample, and thus represented the uncertainty in the MAE! Why is the MAE uncertain?

<h4>PROBLEM</h4>

<strong>Use the </strong><strong>quantile()</strong><strong> function to print out summary statistics associated with the MAE. You can use the default arguments, and thus pass in </strong><strong>MAE_01</strong><strong> into </strong><strong>quantile()</strong><strong> without setting any other argument. Why is the MAE uncertain? Or put another way, what causes the MAE to be uncertain?</strong>

<h4>SOLUTION</h4>

Calculate the quantiles of the MAE below.

###

?

<h2>Problem 03</h2>

You will now make use of the model fitting and prediction functions you created in the previous problems to study the behavior of a more complicated non-linear modeling task. The code chunk below reads in two data sets. Both consist of two continuous variables, an input x and a response y. The first, train_02, will serve as the training set, and the second, test_02, will serve as a hold-out test set. You will only fit models based on the training set.

train_02 &lt;- readr::read_csv(“https://raw.githubusercontent.com/jyurko/CS_1675_Spring_2020/master/hw_data/hw04/train_02.csv”, col_names = TRUE)## Parsed with column specification:## cols(##   x = col_double(),##   y = col_double()## )test_02 &lt;- readr::read_csv(“https://raw.githubusercontent.com/jyurko/CS_1675_Spring_2020/master/hw_data/hw04/test_02.csv”, col_names = TRUE)## Parsed with column specification:## cols(##   x = col_double(),##   y = col_double()## )

<h3>3a)</h3>

It’s always a good idea to start out by visualizing the data before modeling.

<h4>PROBLEM</h4>

<strong>Create a scatter plot between the response and the input with </strong><strong>ggplot2</strong><strong>. Include both the training and test sets together in one graph. Use the marker color to distinguish between the two.</strong>

<h4>SOLUTION</h4>

###

<h3>3b)</h3>

You will fit 25 different models to the training set. You will consider a first-order spline up to a 25th order spline. Your goal will be to find which spline is the “best” in terms of generalizing from the training set to the test set. To do so, you will calculate the MAE on the training set and on the test set for each model. It will be rather tedious to set up all of the necessary information by hand, manually train each model, generate posterior samples, and make predictions from each model. Therefore, you will work through completing functions that will enable you to programmatically loop over each candidate model.

You will start out by completing a function to create the training set and test set for a desired spline basis. The function make_spline_basis_mats() is started for you in the first code chunk below. The first argument is the desired spline basis order, J. The second argument is the training set, train_data, and the third argument is the hold-out test set, test_data.

The second through fifth code chunks below are provided to check that you completed the function correctly. The second code chunk uses purrr::map_dfr() to create the training and test matrices for all 25 models. A glimpse of the resulting object, spline_matrices, is displayed to the screen for you in the third code chunk. It is printed to the screen in the fourth code chunk. You should see a tibble consisting of two variables, design_matrix and test_matrix. Both variables are lists containing matrices. The matrices contained in the spline_matrices$design_matrix variable are the different training design matrices, while the matrices contained in spline_matrices$test_matrix are the associated hold-out test basis matrices.

The fifth code chunk below prints the dimensionality of the 1st and 2nd order spline basis matrices to the screen. It shows that to access a specific matrix, you need to use the [[]] notation.

<h4>PROBLEM</h4>

<strong>Complete the code chunk below. You must specify the </strong><strong>splines::ns()</strong><strong> function call correctly such that the degrees-of-freedom, </strong><strong>df</strong><strong> argument equals the desired spline basis order and that the basis is applied to the </strong><strong>x</strong><strong> variable within the user supplied </strong><strong>train_data</strong><strong> argument. The knots are extracted for you and saved to the </strong><strong>knots_use_basis</strong><strong> object. Create the training design matrix by calling the </strong><strong>model.matrix()</strong><strong> function with the </strong><strong>splines::ns()</strong><strong> function to create the basis for the </strong><strong>x</strong><strong> variable with </strong><strong>knots</strong><strong> equal to </strong><strong>knots_use_basis</strong><strong>. Make sure you assign the data sets correctly to the </strong><strong>data</strong><strong> argument of </strong><strong>model.matrix()</strong><strong>.</strong>

<strong>How many rows are in the training matrices and how many rows are in the test matrices?</strong>

<h4>SOLUTION</h4>

Define the make_spline_basis_mats() function.

make_spline_basis_mats &lt;- function(J, train_data, test_data){  train_basis &lt;- splines::ns( )    knots_use_basis &lt;- as.vector(attributes(train_basis)$knots)    train_matrix &lt;- model.matrix( )    test_matrix &lt;- model.matrix( )    tibble::tibble(    design_matrix = list(train_matrix),    test_matrix = list(test_matrix)  )}

Create each of the training and test basis matrices.

spline_matrices &lt;- purrr::map_dfr(1:25, make_spline_basis_mats,                                  train_data = train_02,                                   test_data = test_02)

Get a glimpse of the structure of spline_matrices.

glimpse(spline_matrices)

Display the elements of spline_matrices to the screen.

spline_matrices

Check the dimensionality of several training and test matrices.

dim(spline_matrices$design_matrix[[1]]) dim(spline_matrices$test_matrix[[1]]) dim(spline_matrices$design_matrix[[2]]) dim(spline_matrices$test_matrix[[2]])

?

<h3>3c)</h3>

Each element in the spline_matrices$design_matrix object is a separate design matrix. You will use this structure to programmatically train each model. The first code chunk creates a list of information which stores the training set responses and defines the prior hyperparameters. The second code chunk below defines the manage_spline_fit() function. The first argument is a design matrix Xtrain, the second argument is the log-posterior function, logpost_func, and the third argument is my_settings. manage_spline_fit() sets the initial starting values to the (boldsymbol{beta}) parameters by generating random values from a standard normal. The initial value for the unbounded (varphi) parameter is set by log-transforming a random draw from the prior on (sigma). After creating the initial guess values, the my_laplace() function is called to fit the model.

You will complete both code chunks in order to programmatically train all 25 spline models. After completing the first two code chunks, the third code chunk performs the training for you. The fourth code chunk below shows how to access training results associated with the second-order spline by using the [[]] operator. The fifth code chunk checks that each model converged.

<h4>PROBLEM</h4>

<strong>Complete the first two code chunks below. In the first code chunk, assign the training responses to the </strong><strong>yobs</strong><strong> variable within the </strong><strong>info_02_train</strong><strong> list. Specify the prior mean and prior standard deviation on the </strong><strong>(boldsymbol{beta})</strong><strong>-parameters to be 0 and 20, respectively. Specify the rate parameter on the unknown </strong><strong>(sigma)</strong><strong> to be 1.</strong>

<strong>Complete the second code chunk by generating a random starting guess for all </strong><strong>(boldsymbol{beta})</strong><strong>-parameters from a standard normal. Create the random initial guess for </strong><strong>(varphi)</strong><strong> by generating a random number from the Exponential prior distribution on </strong><strong>(sigma)</strong><strong> and log-transforming the variable. Complete the call the </strong><strong>my_laplace()</strong><strong> function by passing in the initial values as a vector of correct size.</strong>

<em>HINT</em>: How can you determine the number of unknown (boldsymbol{beta})-parameters if you know the training design matrix?

<h4>SOLUTION</h4>

Assemble the list of required information.

info_02_train &lt;- list(  yobs = ,  mu_beta = ,  tau_beta = ,  sigma_rate = )

Complete the function which manages the execution of the Laplace Approximation to each spline model.

manage_spline_fit &lt;- function(Xtrain, logpost_func, my_settings){  my_settings$design_matrix &lt;-     init_beta &lt;-     init_varphi &lt;-     my_laplace( )}

Train all 25 spline models. Notice that because the training design matrices have already been created, we just need to loop over each element of spline_matrices$design_matrix.

set.seed(724412)all_spline_models &lt;- purrr::map(spline_matrices$design_matrix,                                 manage_spline_fit,                                logpost_func = lm_logpost,                                my_settings = info_02_train)

Check the Laplace Approximation results of the second order spline.

all_spline_models[[2]]

Check that the optimizations successfully converged for each model.

purrr::map_chr(all_spline_models, “converge”)

<h3>3d)</h3>

With all 25 spline models fit, it is time to assess which model is the best. Several different approaches have been discussed in lecture for how to identify the “best” model. You will start out by calculating the MAE on the training set and the test set. You went through the steps to generate posterior samples, make posterior predictions and to calculate the posterior MAE distribution in Problem 2. You will now define a function which performs all of those steps together.

The function calc_mae_from_laplace() is started for you in the first code chunk below. The first argument, mvn_result, is the result of the my_laplace() function for a particular model. The second and third arguments, Xtrain and Xtest, are the training and test basis matrices associated with the model, respectively. The fourth and fifth arguments, y_train and y_test, are the observations on the training set and test set, respectively. The last argument, num_samples, is the number of posterior samples to generate.

You will complete the necessary steps to generate posterior samples from the model, predict the training set, predict the test. Then you will calculate the training set MAE and test set MAE, associated with each posterior sample. The last portion of the calc_mae_from_laplace() function is mostly completed for you.

After you complete the calc_mae_from_laplace(), the second code chunk applies the function to all 25 spline models. It is nearly complete. You must specify the arguments which define the training set observations, y_train, the test set observations, y_train, and the number of posterior samples, num_samples.

<h4>PROBLEM</h4>

<strong>Complete all steps to calculate the MAE on the training set and test sets in the first code chunk below. Complete the lines of code in order to: generate posterior samples from the supplied </strong><strong>mvn_result</strong><strong> object, make posterior predictions on the training set, make posterior predictions on the test, and then calculate the MAE associated with each posterior sample on the training and test sets. In the book keeping portion of the function, you must specify the order of the spline model.</strong>

<strong>You must specify the training set and test set observed responses correctly in the second code chunk. You must specify the number of posterior samples to be 2500.</strong>

<em>HINT</em>: Remember that the result of the make_post_lm_pred() function is a list!

<h4>SOLUTION</h4>

Complete all steps to define the calc_mae_from_laplace() function below.

calc_mae_from_laplace &lt;- function(mvn_result, Xtrain, Xtest, y_train, y_test, num_samples){  # generate posterior samples from the approximate MVN posterior  post &lt;-     # make posterior predictions on the training set  pred_train &lt;-     # make posterior predictions on the test set  pred_test &lt;-     # calculate the error between the training set predictions  # and the training set observations  error_train &lt;-     # calculate the error between the test set predictions  # and the test set observations  error_test &lt;-     # calculate the MAE on the training set  mae_train &lt;-     # calculate the MAE on the test set  mae_test &lt;-     # book keeping, package together the results  mae_train_df &lt;- tibble::tibble(    mae = mae_train  ) %&gt;%     mutate(dataset = “training”) %&gt;%     tibble::rowid_to_column(“post_id”)    mae_test_df &lt;- tibble::tibble(    mae = mae_test  ) %&gt;%     mutate(dataset = “test”) %&gt;%     tibble::rowid_to_column(“post_id”)    # you must specify the order, J, associated with the spline model  mae_train_df %&gt;%     bind_rows(mae_test_df) %&gt;%     mutate(J = )}

Apply the calc_mae_from_laplace() function to all 25 spline models.

set.seed(52133)all_spline_mae_results &lt;- purrr::pmap_dfr(list(all_spline_models,                                               spline_matrices$design_matrix,                                               spline_matrices$test_matrix),                                          calc_mae_from_laplace,                                          y_train = ,                                          y_test = ,                                          num_samples = )

<h3>3e)</h3>

If you completed the calc_mae_from_laplace() function correctly, you should have an object with 125000 rows and just 4 columns. The object was structured in a “tall” or “long-format” in order to allow summarizing all of the posterior MAE samples with ggplot2. You will summarize the MAE posterior distributions with boxplots and by focusing on the median MAE. To focus on the median MAE, you will use the stat_summary() function. This is a flexible function capable of creating many different geometric objects. It consists of arguments such as geom to specify the type of geometric object to display and fun.y to specify what function to apply to the y aesthetic.

<h4>PROBLEM</h4>

<strong>Complete the two code chunks below. In the first code chunk, pipe the </strong><strong>all_spline_mae_results</strong><strong> object into </strong><strong>ggplot()</strong><strong>. Set the </strong><strong>x</strong><strong> aesthetic to be </strong><strong>as.factor(J)</strong><strong> and the </strong><strong>y</strong><strong> aesthetic to be </strong><strong>mae</strong><strong>. In the </strong><strong>geom_boxplot()</strong><strong> call, map the </strong><strong>fill</strong><strong> aesthetic to the </strong><strong>dataset</strong><strong> variable. Use the scale </strong><strong>scale_fill_brewer()</strong><strong> with the </strong><strong>palette</strong><strong> argument set equal to </strong><strong>“Set1”</strong><strong>.</strong>

<strong>In the second code chunk, pipe the </strong><strong>all_spline_mae_results</strong><strong> object into a </strong><strong>filter()</strong><strong> call. Keep only the models with the spline order greater than 3. Pipe the result into a </strong><strong>ggplot()</strong><strong> call where you set the </strong><strong>x</strong><strong> aesthetic to </strong><strong>J</strong><strong> and the </strong><strong>y</strong><strong> aesthetic to </strong><strong>mae</strong><strong>. Use the </strong><strong>stat_summary()</strong><strong> call to calculate the median MAE for each spline order. You must set the </strong><strong>geom</strong><strong> argument within </strong><strong>stat_summary()</strong><strong> to be </strong><strong>‘line’</strong><strong> and the </strong><strong>fun.y</strong><strong> argument to be </strong><strong>‘median’</strong><strong>. Rather than coloring by </strong><strong>dataset</strong><strong>, use </strong><strong>facet_wrap()</strong><strong> to specify separate facets (subplots) for each </strong><strong>dataset</strong><strong>.</strong>

<strong>Based on these visualizations which models appear to overfit the training data?</strong>

<h4>SOLUTION</h4>

Summarize the posterior samples on the MAE on the training and test sets with boxplots.

###

Summarize the posterior samples on the MAE on the training and test sets by focusing on the posterior median MAE values.

###

?

<h3>3f)</h3>

By comparing the posterior MAE values on the training and test splits you are trying to assess how well the models generalize to new data. As discussed in lecture, other metrics exist for trying to assess how well a model generalizes, based just on the training set performance. The Evidence or marginal likelihood is attempting to evaluate generalization by integrating the likelihood over all a-priori allowed parameter combinations. If the Evidence can be calculated, it can be used to weight all models relative to each other. Thereby allowing you to assess which model appears to be “most probable”.

You will now calculate the posterior model weights associated with each model. The first code chunk below is completed for you, by extracting the log-evidence associated with model into the numeric vector spline_evidence. You will use the log-evidence to calculate the posterior model weights and visualize the results with a bar graph.

<h4>PROBLEM</h4>

<strong>Calculate the posterior model weights associated with each spline model and assign the weights to the </strong><strong>spline_weights</strong><strong> variable. The </strong><strong>spline_weights</strong><strong> vector is assigned to the </strong><strong>w</strong><strong> variable in a </strong><strong>tibble</strong><strong> and the result is piped into </strong><strong>tibble::rowid_to_column()</strong><strong> where the row identification label is named </strong><strong>J</strong><strong> to correspond to the spline basis order. Pipe the result into a </strong><strong>ggplot()</strong><strong> call where you set the </strong><strong>x</strong><strong> aesthetic to </strong><strong>as.factor(J)</strong><strong> and the </strong><strong>y</strong><strong> aesthetic to </strong><strong>w</strong><strong>. Include a </strong><strong>geom_bar()</strong><strong> geometric object where the </strong><strong>stat</strong><strong> argument is set to </strong><strong>“identity”</strong><strong>.</strong>

<strong>Based on your visualization, which model is considered the best?</strong>

spline_evidence &lt;- purrr::map_dbl(all_spline_models, “log_evidence”)spline_weights &lt;-  tibble::tibble(  w = ) %&gt;%   tibble::rowid_to_column(“J”) %&gt;%   ggplot()

?

<h3>3g)</h3>

You have compared the models several different ways, are your conclusions the same?

<h4>PROBLEM</h4>

<strong>How well do the assessments from the data split comparison compare to the Evidence-based assessment? If the conclusions are different, why would they be different?</strong>

<h4>SOLUTION</h4>

?

<h2>Problem 04</h2>

In lecture we discussed how basis functions allow extending the linear model to handle non-linear relationships. It was also discussed how to generalize the linear modeling approach to binary outcomes with logistic regression. In this problem you will define the log-posterior function for logistic regression. By doing so, you will be able to directly contrast what you did to define the log-posterior function for the linear model in previous problems within this assignment.

<h3>4a)</h3>

The complete probability model for logistic regression consists of the likelihood of the response (y_n) given the event probability (mu_n), the inverse link function between the probability of the event, (mu_n), and the linear predictor, (eta_n), and the prior on all linear predictor model coefficients (boldsymbol{beta}).

As in lecture, you will assume that the (boldsymbol{beta})-parameters are a-priori independent Gaussians with a shared prior mean (mu_{beta}) and a shared prior standard deviation (tau_{beta}).

<h4>PROBLEM</h4>

<strong>Write out complete probability model for logistic regression. You must write out the </strong><strong>(n)</strong><strong>-th observation’s linear predictor using the inner product of the </strong><strong>(n)</strong><strong>-th row of a design matrix </strong><strong>(mathbf{x}_{n,:})</strong><strong> and the unknown </strong><strong>(boldsymbol{beta})</strong><strong>-parameter column vector. You can assume that the number of unknown coefficients is equal to </strong><strong>(D + 1)</strong><strong>.</strong>

You are allowed to separate each equation into its own equation block.

<em>HINT</em>: The “given” sign, the vertical line, (mid), is created by typing mid in a latex math expression. The product symbol (the giant PI) is created with prod_{}^{}.

<h4>SOLUTION</h4>

?

<h3>4b)</h3>

The code chunk below loads in a data set consisting of two variables. A continuous input x and a binary outcome y. The binary outcome is encoded as 0 if the event does not occur and 1 if the event does occur. The count() function is used to count the number of observations associated with each binary class in the second code chunk. As shown below, the classes are roughly balanced.

train_03 &lt;- readr::read_csv(“https://raw.githubusercontent.com/jyurko/CS_1675_Spring_2020/master/hw_data/hw04/train_03.csv”, col_names = TRUE)## Parsed with column specification:## cols(##   x = col_double(),##   y = col_double()## )train_03 %&gt;% glimpse()## Observations: 100## Variables: 2## $ x &lt;dbl&gt; -0.99886904, -1.35011298, 0.95776645, -0.60482885, 1.9026864…## $ y &lt;dbl&gt; 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, …train_03 %&gt;% count(y)## # A tibble: 2 x 2##       y     n##   &lt;dbl&gt; &lt;int&gt;## 1     0    54## 2     1    46

You will fit two logistic regression models to the dataset. The first will be a linear relationship for the linear predictor and the second will be a cubic relationship. The linear predictor expressions are written for you in the equation blocks below.

[ eta_n = beta_0 + beta_1 x_n ]

[ eta_n = beta_0 + beta_1 x_n + beta_2 x_{n}^2 + beta_3 x_{n}^3 ]

Before creating the log-posterior function for logistic regression, you will create lists of necessary information in the same style that you did in the previous regression problems. You must create the design matrices associated with the linear and cubic relationships, Xmat_03_line and Xmat_03_cube. You must then complete the lists info_03_line and info_03_cube by setting the observed responses to the yobs variable, the design matrices to the design_matrix variable, and also set the (boldsymbol{beta}) prior hyperparameters.

<h4>PROBLEM</h4>

<strong>Create the design matrices for the linear and cubic relationships. Specify the lists of required information for both models. Specify the prior mean to be 0 and the prior standard deviation to be 5.</strong>

<h4>SOLUTION</h4>

Create each design matrix.

Xmat_03_line &lt;-  Xmat_03_cube &lt;-

Create the lists of required information.

info_03_line &lt;- list(  yobs = ,  design_matrix = ,  mu_beta = ,  tau_beta = ) info_03_cube &lt;- list(  yobs = ,  design_matrix = ,  mu_beta = ,  tau_beta = )

<h3>4c)</h3>

You will now define the log-posterior function for logistic regression, glm_logpost(). The first argument to glm_logpost() is the vector of unknowns and the second argument is the list of required information. You will assume that the variables within the my_info list are those contained in the info_03_line list you defined previously.

<h4>PROBLEM</h4>

<strong>Complete the code chunk to define the </strong><strong>glm_logpost()</strong><strong> function. The comments describe what you need to fill in. Do you need to separate out the </strong><strong>(boldsymbol{beta})</strong><strong>-parameters from the vector of </strong><strong>unknowns</strong><strong>?</strong>

<strong>After you complete </strong><strong>glm_logpost()</strong><strong>, test it by setting the </strong><strong>unknowns</strong><strong> vector to be a vector of -1’s for the linear relationship case. If you have succesfully programmed the function you should get a result equal to </strong><strong>-118.3933</strong><strong>.</strong>

<h4>SOLUTION</h4>

?

glm_logpost &lt;- function(unknowns, my_info){  # extract the design matrix and assign to X  X &lt;-     # calculate the linear predictor  eta &lt;-     # calculate the event probability  mu &lt;-     # evaluate the log-likelihood  log_lik &lt;-     # evaluate the log-prior  log_prior &lt;-     # sum together  }

Test out your function using the linear relationship information and setting the unknowns to a vector of -1’s.

glm_logpost( )

<h3>4d)</h3>

Execute the Laplace Approximation for the linear and cubic models. Use initial guess values of zero for both models. After fitting both models, calculate the middle 95% uncertainty intervals for the cubic model parameters.

<h4>PROBLEM</h4>

<strong>Perform the Laplace Approximation for the linear and cubic models. Set the initial guesses to vectors of zero for both models. Should you be concerned that the initial guess will impact the results?</strong>

<strong>After fitting, calculate the middle 95% uncertainty interval on the </strong><strong>(boldsymbol{beta})</strong><strong> parameters for both models. Which parameters contain zero in the middle 95% uncertainty interval?</strong>

<h4>SOLUTION</h4>

?

laplace_03_line &lt;-  laplace_03_cube &lt;-

Calculate the posterior middle 95% uncertainty interval on the parameters associated with each model.

###

<h3>4e)</h3>

Let’s compare the performance of the models using the Evidence-based assessment. Since we only have two models, calculate the Bayes Factor between the linear model and the cubic model. Based on the Bayes Factor which model do you think is better?

<h4>PROBLEM</h4>

<strong>Calculate the Bayes Factor with the linear model in the numerator and the cubic model in the denominator. Which model is supported more from the data?</strong>

<h4>SOLUTION</h4>

###

?

<h3>4f)</h3>

You will now spend a little more time with the cubic relationship results. Calculate the posterior correlation matrix between the unknown (boldsymbol{beta}) parameters. Are any of the parameter highly correlated or anti-correlated?

<h4>PROBLEM</h4>

<strong>Calculate the posterior correlation matrix between the parameters in the cubic model.</strong>

<h4>SOLUTION</h4>

?

###

<h2>Problem 05</h2>

In Problem 04, you compared the linear and cubic relationships based on the Evidence. Your assessment considered how well the model “fit” the data via the likelihood, based on the constraints imposed by the prior. The likelihood examines how likely the binary outcome is given the event probability. Thus, the Evidence is considering if the observations are consistent with the modeled event probability. In this problem, you will consider point-wise error metrics by calculating the confusion matrix associated with the training set. Confusion matrices are useful because the accuracy and errors are in the same “units” of the data.

However, the logistic regression model predicts the event probability via the log-odds ratio. In order to move from the probability to the binary outcome a decision must be made. As discussed during the Applied Machine Learning portion of the course, the decision consists of comparing the predicted probability to a threshold value. If the predicted probability is greater than the threshold, classify the outcome as the event. Otherwise, classify the outcome as the non-event.

In order to classify the training points, you must make posterior predictions with the logistic regression models you fit in Problem 04.

<h3>5a)</h3>

Although you were able to apply the my_laplace() function to both the regression and logistic regression settings, you cannot directly apply the generate_lm_post_samples() function. You will therefore adapt generate_lm_post_samples() and define generate_glm_post_samples(). The code chunk below starts the function for you and uses just two input arguments, mvn_result and num_samples. You must complete the function.

<h4>PROBLEM</h4>

<strong>Why can you not directly use the </strong><strong>generate_lm_post_samples()</strong><strong> function? Since the </strong><strong>length_beta</strong><strong> argument is NOT provided to </strong><strong>generate_glm_post_samples()</strong><strong>, how can you determine the number of </strong><strong>(boldsymbol{beta})</strong><strong>-parameters? Complete the code chunk below by first assigning the number of </strong><strong>(boldsymbol{beta})</strong><strong>-parameters to the </strong><strong>length_beta</strong><strong> variable. Then generate the random samples from the MVN distribution. You do not have to name the variables, you only need to call the correct random number generator.</strong>

<h4>SOLUTION</h4>

?

generate_glm_post_samples &lt;- function(mvn_result, num_samples){  # specify the number of unknown beta parameters  length_beta &lt;-     # generate the random samples  beta_samples &lt;-     # change the data type and name  beta_samples %&gt;%     as.data.frame() %&gt;% tbl_df() %&gt;%     purrr::set_names(sprintf(“beta_%02d”, (1:length_beta) – 1))}

<h3>5b)</h3>

You will now define a function which calculates the posterior prediction samples on the linear predictor and the event probability. The function, post_glm_pred_samples() is started for you in the code chunk below. It consists of two input arguments Xnew and Bmat. Xnew is a test design matrix where rows correspond to prediction points. The matrix Bmat stores the posterior samples on the (boldsymbol{beta})-parameters, where each row is a posterior sample.

<h4>PROBLEM</h4>

<strong>Complete the code chunk below by using matrix math to calculate the linear predictor at every posterior sample. Then, calculate the event probability for every posterior sample.</strong>

The eta_mat and mu_mat matrices are returned within a list, similar to how the Umat and Ymat matrices were returned for the regression problems.

<em>HINT</em>: The boot::inv.logit() can take a matrix as an input. When it does, it returns a matrix as a result.

<h4>SOLUTION</h4>

post_glm_pred_samples &lt;- function(Xnew, Bmat){  # calculate the linear predictor at all prediction points and posterior samples  eta_mat &lt;-     # calculate the event probability  mu_mat &lt;-    # book keeping  list(eta_mat = eta_mat, mu_mat = mu_mat)}

<h3>5c)</h3>

The code chunk below defines a function summarize_glm_pred_from_laplace() which manages the actions necessary to summarize posterior predictions of the event probability. The first argument, mvn_result, is the Laplace Approximation object. The second object is the test design matrix, Xtest, and the third argument, num_samples, is the number of posterior samples to make. You must follow the comments within the function in order to generate posterior prediction samples of the linear predictor and the event probability, and then to summarize the posterior predictions of the event probability.

The result from summarize_glm_pred_from_laplace() summarizes the posterior predicted event probability with the posterior mean, as well as the 5th and 95th quantiles. If you have completed the post_glm_pred_samples() function correctly, the dimensions of the mu_mat matrix should be consistent with those from the Umat matrix from the regression problems. The posterior summary statistics summarize over all posterior samples. You must therefore choose between colMeans() and rowMeans() as to how to calculate the posterior mean event probability for each prediction point. The posterior quantiles are calculated for you.

<h4>PROBLEM</h4>

<strong>Follow the comments in the code chunk below to complete the definition of the </strong><strong>summarize_glm_pred_from_laplace()</strong><strong> function. You must generate posterior samples, make posterior predictions, and then </strong>

<em>HINT</em>: The result from post_glm_pred_samples() is a list.

<h4>SOLUTION</h4>

summarize_glm_pred_from_laplace &lt;- function(mvn_result, Xtest, num_samples){  # generate posterior samples of the beta parameters  betas &lt;-     # data type conversion  betas &lt;- as.matrix(betas)    # make posterior predictions on the test set  pred_test &lt;-     # calculate summary statistics on the posterior predicted probability  # summarize over the posterior samples    # posterior mean, should you summarize along rows (rowMeans) or   # summarize down columns (colMeans) ???  mu_avg =     # posterior quantiles  mu_q05 = apply(pred_test$mu_mat, 1, stats::quantile, probs = 0.05)  mu_q95 = apply(pred_test$mu_mat, 1, stats::quantile, probs = 0.95)    # book keeping  tibble::tibble(    mu_avg = mu_avg,    mu_q05 = mu_q05,    mu_q95 = mu_q95  ) %&gt;%     tibble::rowid_to_column(“pred_id”)}

<h3>5d)</h3>

Summarize the posterior predicted event probability associated with the training set for both the linear and cubic relationships. After making the predictions, a code chunk is provided for you which generates a figure showing how the posterior predicted probability summaries compare with the observed binary outcomes. Which of the two models appears to capture the trends in the binary outcomes better?

<h4>PROBLEM</h4>

<strong>Call </strong><strong>summarize_glm_pred_from_laplace()</strong><strong> for the linear and cubic relationships on the training set. Specify the number of posterior samples to be 2500. Print the dimensions of the resulting objects to the screen. How many rows are in each data set?</strong>

<strong>The third code chunk below uses the prediction summaries to visualize the posterior predicted event probability on the training set. Which relationship seems more in line with the observations?</strong>

<h4>SOLUTION</h4>

Execute the prediction summaries in the code chunk below.

post_pred_summary_03_line &lt;-  post_pred_summary_03_cube &lt;-

Print the dimensions of the objects to the screen.

###

The figure below is created for you. The posterior predicted mean event probabilities are displayed by the navyblue curves. The posterior predicted middle 90% uncertainty intervals on the event probabilities are shown by the light blue ribbon.

post_pred_summary_03_line %&gt;%   mutate(type = “linear relationship”) %&gt;%   bind_rows(post_pred_summary_03_cube %&gt;%               mutate(type = “cubic relationship”)) %&gt;%   left_join(train_03 %&gt;% tibble::rowid_to_column(“pred_id”),            by = “pred_id”) %&gt;%   ggplot(mapping = aes(x = x)) +  geom_ribbon(mapping = aes(ymin = mu_q05,                            ymax = mu_q95,                            group = type),              fill = “steelblue”, alpha = 0.5) +  geom_line(mapping = aes(y = mu_avg,                          group = type),            color = “navyblue”, size = 1.15) +  geom_point(mapping = aes(y = y),             size = 2.5, alpha = 0.25) +  facet_grid( . ~ type) +  labs(y = “y or probability”) +  theme_bw()

?

<h3>5e)</h3>

You will now consider classifying the predictions based upon a threshold value of 0.5. You will compare that threshold value to the posterior predicted event probabilities associated with the training set. Although the Bayesian model provides a full posterior predictive distribution, you will work just with the posterior mean value. Thus, you will create a single confusion matrix, rather than considering the uncertainty in the confusion matrix.

Creating the confusion matrix is rather simple compared to some of the previous tasks in this assignment. The first step is to classify the prediction as event or non-event, which can be accomplished with an if-statement. The ifelse() function provides an “Excel-like” conditional statement, and is a simple way to perform the classification task. The syntax for ifelse() consists of three arguments, shown below:

ifelse(&lt;conditional test&gt;, &lt;return if condition is TRUE&gt;, &lt;return if condition is FALSE&gt;)

The first argument is the conditional test you wish to apply. The second argument is what will be returned if the condition is true, and the third argument is what will be returned if the condition is false.

You will use the ifelse() function to compare the posterior predicted mean event probability to the assumed threshold value of 0.5.

<h4>PROBLEM</h4>

<strong>Pipe the </strong><strong>post_pred_summary_03_line</strong><strong> object into a </strong><strong>mutate()</strong><strong> call and create a new variable </strong><strong>pred_y</strong><strong> which is the result of an </strong><strong>ifelse()</strong><strong> operation. For the conditional test, return a value of </strong><strong>1</strong><strong> if the posterior predicted mean event probability is greater than 0.5, and return </strong><strong>0</strong><strong> otherwise. Repeat the process for the </strong><strong>post_pred_summary_03_cube</strong><strong> object.</strong>

<h4>SOLUTION</h4>

post_pred_summary_03_line_b &lt;-  post_pred_summary_03_cube_b &lt;-

<h3>5f)</h3>

The code chunk below uses the left_join() function to merge the training data set, train_03 with each of the posterior prediction summary objects. The results, post_pred_summary_03_line_c and post_pred_summary_03_cube_c both now have predicted classifications, pred_y, and observed outcomes y.

post_pred_summary_03_line_c &lt;- post_pred_summary_03_line_b %&gt;%   left_join(train_03 %&gt;% tibble::rowid_to_column(“pred_id”),            by = “pred_id”) post_pred_summary_03_cube_c &lt;- post_pred_summary_03_cube_b %&gt;%   left_join(train_03 %&gt;% tibble::rowid_to_column(“pred_id”),            by = “pred_id”)

You now have everything you need to calculate the confusion matrix for the linear and cubic relationship models. A simple way to do this is with the count() function from dplyr, which counts the unique combinations of the variables you provide to it. count() returns a data.frame with the combination of the columns used to perform the grouping and counting operation, as well as a new column n which stores the number of rows associated with each combination.

<h4>PROBLEM</h4>

<strong>Use the </strong><strong>count()</strong><strong> function to determine the confusion matrix associated with each relationship. How many true-positives, true-negatives, false-positives, and false-negatives does each relationship have?</strong>

<h4>SOLUTION</h4>

?

### confusion matrix for the linear relationship

?

### confusion matrix for the cubic relationshiop


