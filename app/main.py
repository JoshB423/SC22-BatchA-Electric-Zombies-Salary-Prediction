<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
        <meta name="description" content="" />
        <meta name="author" content="" />
        <title>Salary Prediction</title>
        <!-- Favicon-->
        <link rel="icon" type="image/x-icon" href="https://aux3.iconspalace.com/uploads/543120981.png" />
        <!-- Font Awesome icons (free version)-->
        <script src="https://use.fontawesome.com/releases/v6.1.0/js/all.js" crossorigin="anonymous"></script>
        <!-- Google fonts-->
        <link href="https://fonts.googleapis.com/css?family=Montserrat:400,700" rel="stylesheet" type="text/css" />
        <link href="https://fonts.googleapis.com/css?family=Roboto+Slab:400,100,300,700" rel="stylesheet" type="text/css" />
        <!-- Core theme CSS (includes Bootstrap)-->
        <link href="static/css/styles.css" rel="stylesheet" />

        <!-- To automatically render math in text elements, include the auto-render extension: -->
        <script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.0-rc.1/dist/contrib/auto-render.min.js" integrity="sha384-yACMu8JWxKzSp/C1YV86pzGiQ/l1YUfE8oPuahJQxzehAjEt2GiQuy/BIvl9KyeF" crossorigin="anonymous"
          onload="renderMathInElement(document.body);">
        </script>

        <!-- The loading of KaTeX is deferred to speed up page rendering -->
        <script src="https://cdn.jsdelivr.net/npm/katex@0.10.0-rc.1/dist/katex.min.js" integrity="sha384-483A6DwYfKeDa0Q52fJmxFXkcPCFfnXMoXblOkJ4JcA8zATN6Tm78UNL72AKk+0O" crossorigin="anonymous"></script>


        <!-- Math -->
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.10.0-rc.1/dist/katex.min.css" integrity="sha384-D+9gmBxUQogRLqvARvNLmA9hS2x//eK1FhVb9PiU86gmcrBrJAQT8okdJ4LMp2uv" crossorigin="anonymous">


    </head>
    <body id="page-top">
        <!-- Navigation-->
      
        <nav class="navbar navbar-expand-lg navbar-dark fixed-top" id="mainNav">
            <div class="container">
                <a class="navbar-brand" href="#page-top">
                  <img src="https://cdn.discordapp.com/attachments/788331009020002315/989273334996955216/Untitled_Artwork.png" width = auto height = "500" alt="..." /></a>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarResponsive" aria-controls="navbarResponsive" aria-expanded="false" aria-label="Toggle navigation">
                    Menu
                    <i class="fas fa-bars ms-1"></i>
                </button>
                <div class="collapse navbar-collapse" id="navbarResponsive">
                    <ul class="navbar-nav text-uppercase ms-auto py-4 py-lg-0">
                        <li class="nav-item"><a class="nav-link" href="#services">About</a></li>
                       <li class="nav-item"><a class="nav-link" href="#DataCleansing">Data Cleansing</a></li>
                        <li class="nav-item"><a class="nav-link" href="#portfolio">EDA</a></li>
                        <li class="nav-item"><a class="nav-link" href="#about">Pipeline</a></li>
                        <li class="nav-item"><a class="nav-link" href="#mlmodels">Machine Learning Models</a></li>
                        <li class="nav-item"><a class="nav-link" href="#team">Team</a></li>
                    </ul>
                </div>
            </div>
        </nav>
        <!-- Masthead-->
        <header class="masthead">
            <div class="container">
                <div class="masthead-subheading">Electric Zombies Presents</div>
                <div class="masthead-heading text-uppercase" >Salary Predictions</div>
                <a class="btn btn-primary btn-xl text-uppercase" href="#services">Tell Me More</a>
            </div>
        </header>
        <!-- Services-->
        <section class="page-section" id="services">
            <div class="container">
                <div class="text-center">
                    <h2 class="section-heading text-uppercase">About</h2>
                    <h3 class="section-subheading text-muted">.</h3>
                </div>
                <div class="row text-center">
                    <div class="col-md-4">

                        <h4 class="my-3">What's this about?</h4>
                        <p class="text-muted">This project is a a predictive algorithm used to predict the salary of someone with the given attributes about themselves, the outcome will predict whether or not you'll make over, under or exactly 50k</p>
                    </div>
                    <div class="col-md-4">

                        <h4 class="my-3">What is our goal?</h4>
                        <p class="text-muted">We've all been at crossroads when it comes to what we want to pursue; however, our A.I. can provide users with a clear path. Our goal is to guide users to their future career based on what they wish to accomplish with their education. </p>
                    </div>

                    <div class="col-md-4">

                        <h4 class="my-3">What is our Dataset about? </h4>
                        <p class="text-muted"> The data set is a classification on salary that determines whether a person makes less than or equal to 50K or greater than 50K based on their personal attributes.
                    </div>

                    <div class="col-md-4">

                        <h4 class="my-3">Classification or Regression? </h4>
                        <p class="text-muted">Classification refers to a binary set data, aka and off or on. Regression refers to data that is cannot be classified as binary, usually numerical or quantifiable data. Our set is classification because this AI categorizes the data to be either over 50k or under 50k.
                    </div>

                    <div class="col-md-4">

                        <h4 class="my-3">What is our MVP? </h4>
                        <p class="text-muted"> A prediction site that provides users with information in regards to their salary. The data set shows the correlation between education, occupation, and salary and predicts whether someone’s salary will be more or less than 50k based on the given information.
                    </div>

                    <div class="col-md-4">

                        <h4 class="my-3">Who and Why?</h4>
                        <p class="text-muted">We are the Electric Zombies, a group of six students attending an A.I. summer camp. One of our main projects in this camp is creating an A.I. model based on data sets. We decided to commit to a salary predictor as it can be helpful to those who may need a clear outcome towards their desired income based on their personal attributes such as occupation and education level.
                    </div>

                </div>
            </div>
        </section>
        <!-- Data Cleaning-->
      <section class="page-section" id="DataCleansing">
            <div class="container">
                <div class="text-center">
                    <h2 class="section-heading text-uppercase">Data Cleaning</h2>
                    <h3 class="section-subheading text-muted"> Data cleaning is the process of preparing data for future analysis by changing or removing data that is incomplete, incorrect, irrelevant, duplicated, or improperly formatted. Data cleansing ensures quality of the data set and provides better accuracy in predictions. If data is not cleaned thoroughly enough, the accuracy of a model may be negatively impacted. </h3>
                </div>
                <div class="row text-center">
                    <div class="col-md-4">

                        <h4 class="my-3">Removing Unwanted Columns</h4>
                        <p class="text-muted">Because real-world data typically contains too much noise and only a few columns that give useful information, it is recommended that these columns be removed before preforming any data analysis. For our data set, we removed columns that contained other sources of income besides the main occupation in order to ensure our data was accurate.  </p>
                    </div>
                    <div class="col-md-4">

                        <h4 class="my-3">Null Values</h4>
                        <p class="text-muted">Real world data always comes with gaps in the data these are called null values. They can be treated in many ways however, one of the most widely used methods (and the one we used ourselves) is to impute the null values with the mean, median, mode or nearby values. </p>
                    </div>
                    <div class="col-md-4">

                        <h4 class="my-3">Encoding Data</h4>
                        <p class="text-muted">Since new world data contains both numerical and categorical data, these categorical columns are required to be encoded. When fed to a machine learning model, as machine  learning can only understand numeric data, these categorical data values are encoded to zeros and ones. Hence, making them more recognizable to machines. </p>
                    </div>
                </div>
            </div>
        </section>


      <!-- Portfolio Grid-->
        <section class="page-section bg-light" id="portfolio">
           <div class="container">
                <div class="text-center">
                    <h2 class="section-heading text-uppercase">Exploratory Data Analysis</h2>
                    <h3 class="section-subheading text-muted">Exploratory data analysis is important for any business since it allows data scientists to analyze the data before coming to any assumption. It ensures that the results produced are valid and applicable to business outcomes and goals. An EDA is a thorough examination meant to uncover the underlying structure of a data set and is important for a company because it exposes trends, patterns, and relationships that are not readily apparent.</h3>
                </div>
                <div class="row">
                    <div class="col-lg-4 col-sm-6 mb-4">
                        <!-- Portfolio item 1-->
                        <div class="portfolio-item">
                            <a class="portfolio-link" data-bs-toggle="modal" href="#portfolioModal1">
                                <div class="portfolio-hover">
                                    <div class="portfolio-hover-content"><i class="fas fa-plus fa-3x"></i></div>
                                </div>
                                <img class = "img-fluid" src="https://media.istockphoto.com/vectors/casual-man-and-woman-thinking-vector-id1223868514?b=1&k=20&m=1223868514&s=612x612&w=0&h=Ri4ECDboLct8I8KF8y5nMaF72IutiTFAD-4hx5ImB40=" width = "415" height = "315" alt="..." />


                            </a>
                            <div class="portfolio-caption">
                                <div class="portfolio-caption-heading">Gender Distribution</div>
                                <div class="portfolio-caption-subheading text-muted"> <p>
                                   Distribution of gender in >50K and <50K
                                  </p> </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-lg-4 col-sm-6 mb-4">
                        <!-- Portfolio item 2-->
                        <div class="portfolio-item">
                            <a class="portfolio-link" data-bs-toggle="modal" href="#portfolioModal2">
                                <div class="portfolio-hover">
                                    <div class="portfolio-hover-content"><i class="fas fa-plus fa-3x"></i></div>
                                </div>
                                <img class = "img-fluid" src="https://i.guim.co.uk/img/media/a3eee50f0a598d36061f6704a79e6a3483b12ad0/0_160_3500_2099/master/3500.jpg?width=1200&height=900&quality=85&auto=format&fit=crop&s=8a00415d83c0c2fb68746b84af15b812" width = "415" height = "315" alt="..." />
                            </a>
                            <div class="portfolio-caption">
                                <div class="portfolio-caption-heading">Gender Pay Gap</div>
                                <div class="portfolio-caption-subheading text-muted"> Is there gender bias in pay? </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-lg-4 col-sm-6 mb-4">
                        <!-- Portfolio item 3-->
                        <div class="portfolio-item">
                            <a class="portfolio-link" data-bs-toggle="modal" href="#portfolioModal3">
                                <div class="portfolio-hover">
                                    <div class="portfolio-hover-content"><i class="fas fa-plus fa-3x"></i></div>
                                </div>
                                <img class="img-fluid" src="https://i.pinimg.com/originals/d4/f8/0c/d4f80c89d14c8321adaa0cdb941f8f7f.jpg" alt="..." />
                            </a>
                            <div class="portfolio-caption">
                                <div class="portfolio-caption-heading">Race</div>
                                <div class="portfolio-caption-subheading text-muted">Does Race have any impact on salary earned? </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-lg-4 col-sm-6 mb-4 mb-lg-0">
                        <!-- Portfolio item 4-->
                        <div class="portfolio-item">
                            <a class="portfolio-link" data-bs-toggle="modal" href="#portfolioModal4">
                                <div class="portfolio-hover">
                                    <div class="portfolio-hover-content"><i class="fas fa-plus fa-3x"></i></div>
                                </div>
                                <img class="img-fluid" src="https://t3.ftcdn.net/jpg/03/34/72/06/360_F_334720648_NUuOH6y4QoARWthm9b9vDk3gIoqNdqbr.jpg" alt="..." />
                            </a>
                            <div class="portfolio-caption">
                                <div class="portfolio-caption-heading">Age Distribution</div>
                                <div class="portfolio-caption-subheading text-muted">Is age one of the factor in earning salary? </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-lg-4 col-sm-6 mb-4 mb-sm-0">
                        <!-- Portfolio item 5-->
                        <div class="portfolio-item">
                            <a class="portfolio-link" data-bs-toggle="modal" href="#portfolioModal5">
                                <div class="portfolio-hover">
                                    <div class="portfolio-hover-content"><i class="fas fa-plus fa-3x"></i></div>
                                </div>
                                <img class="img-fluid" src="https://cdni.iconscout.com/illustration/premium/thumb/graduation-4036129-3345610.png" alt="..." />
                            </a>
                            <div class="portfolio-caption">
                                <div class="portfolio-caption-heading">Higher Education</div>
                                <div class="portfolio-caption-subheading text-muted">Does higher education translates into higher income? </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-lg-4 col-sm-6">
                        <!-- Portfolio item 6-->
                        <div class="portfolio-item">
                            <a class="portfolio-link" data-bs-toggle="modal" href="#portfolioModal6">
                                <div class="portfolio-hover">
                                    <div class="portfolio-hover-content"><i class="fas fa-plus fa-3x"></i></div>
                                </div>
                                <img class="img-fluid" src="https://cdn.dribbble.com/users/455169/screenshots/12873957/heatmap_layers__animated__4.png?compress=1&resize=400x300" alt="..." />
                            </a>
                            <div class="portfolio-caption">
                                <div class="portfolio-caption-heading"> Correlation Heat Map</div>
                                <div class="portfolio-caption-subheading text-muted">Any correlation between features and salary?</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
        <!-- About-->
        <section class="page-section" id="about">
            <div class="container">
                <div class="text-center">
                    <h2 class="section-heading text-uppercase">Machine Learning Pipeline</h2>
                    <h3 class="section-subheading text-muted">For solving any machine leaning problem , we need to follow certain steps from acquiring the data to building models.On a whole these steps are called as pipeline.</h3>
                </div>
                <ul class="timeline">
                    <li>
                        <div class="timeline-image"><img class="rounded-circle img-fluid" src="static/assets/img/about/1.jpg" alt="..." /></div>
                        <div class="timeline-panel">
                            <div class="timeline-heading">
                                <h4>Data Validation and Feature Engineering</h4>
                                <h4 class="subheading"></h4>
                            </div>
                            <div class="timeline-body"><p class="text-muted">Feature engineering is the process of selecting, manipulating, and transforming raw data into features that can be used in supervised learning. With the help of data validation, we can improve the the accuracy and quality of the source data.</p></div>
                        </div>
                    </li>
                    <li class="timeline-inverted">
                        <div class="timeline-image"><img class="rounded-circle img-fluid" src="static/assets/img/about/2.jpg" alt="..." /></div>
                        <div class="timeline-panel">
                            <div class="timeline-heading">
                                <h4>Test train Split</h4>
                                <h4 class="subheading"></h4>
                            </div>
                            <div class="timeline-body"><p class="text-muted"> The train-test split is used to estimate the performance of machine learning algorithms that are applicable for prediction-based Algorithms/Applications. We use training data to train the machine learning model and test data to evaluate the performance of the model.
</p></div>.
                        </div>
                    </li>
                    <li>
                        <div class="timeline-image"><img class="rounded-circle img-fluid" src="static/assets/img/about/3.jpg" alt="..." /></div>
                        <div class="timeline-panel">
                            <div class="timeline-heading">
                                <h4>Encoding Categorical data</h4>
                                <h4 class="subheading"></h4>
                            </div>
                            <div class="timeline-body"><p class="text-muted">Since real world data contains both numerical and categorical data, these categorical columns are required to be encoded. when fed to a machine learning model, as machine  learning can only understand numeric data, these inserted values are encoded to zeros and ones.
</p></div>
                        </div>
                    </li>
                  <li class="timeline-inverted" >
                        <div class="timeline-image"><img class="rounded-circle img-fluid" src="static/assets/img/about/4.jpg" alt="..." /></div>
                        <div class="timeline-panel">
                            <div class="timeline-heading">
                                <h4>Building Models</h4>
                                <h4 class="subheading"></h4>
                            </div>
                            <div class="timeline-body"><p class="text-muted"> For machine learning, we used three models; Logistic Regression, Random Forest and Neural Network. These models are used for classification of the sample data. After testing each model we found that Neural Network provided the highest accuracy result with 84%.
</p></div>

                            <h4>

                                <br />

                                <br />

                            </h4>
                        </div>
                    </li>
                  <li class="timeline-inverted">
                        <div class="timeline-image">
                            <h4>
                                Model
                                <br />
                                Deployment
                                <br />
                            </h4>
                        </div>
                    </li>
                  
                  
                  
                  
                </ul>
            </div>
        </section>
        <!-- Team-->
        <section class="page-section bg-light" id="mlmodels">
             <div class="container">
                <div class="text-center">
                    <h2 class="section-heading text-uppercase"> Models </h2>
                    <h6 class="section-subheading text-muted"></h6>
                </div>
                <br>
                <div class="row">
                    <div class="col-6 col-sm-4">
                        <div class="team-member">
                            <h4>Logistic Regression </h4>
                            <img  src="https://aigeekprogrammer.com/wp-content/uploads/2019/10/Logistic-Regression-for-binary-classification.jpg" width = "400" height = "400" alt="..." />
                            <p>Logistic regression is a method used to  predict a binary outcome, such as yes or no, based on prior observations of a data set. A Logistic Regression model predicts a dependent data variable by analyzing the relationship between one or more existing variables. In Logistic Regression, an "S" shaped logistic function is used to predict two maximum values (0 or 1). The curve from the Logistic Regression function indicates the likelihood of the occurrence of each data piece.  </p>
                            <br>
                            <p>Confusion Matrix for Logistic Regression:</p>
                          <img src = "static/assets/img/logistic_cm.png" width = "400" height = "400" > 
                            <br>
                          <br>
                          <br>
                            <ul class="list-group">
                              <li class="list-group-item list-group-item-primary">Accuracy : 82% </li>
                              <li class="list-group-item list-group-item-secondary">Precision : 0.87</li>
                              <li class="list-group-item list-group-item-success">Recall : 0.90</li>
                              <li class="list-group-item list-group-item-success">F1-Score : 0.88</li>
                            </ul>
                        </div>
                    </div>
                    <div class="col-6 col-sm-4">
                        <div class="team-member">
                             <h4>Random Forests</h4>
                            <img  src="https://cdn2.hubspot.net/hubfs/4130326/Brendan%20Tierney%20Random%20Forests%201/b2.png" width = "400" height = "400" alt="..." />
                            <p>Random Forests are a sequence of decision trees, that perform classification or regression by asking true or false questions which get solved with a majority vote. Each tree makes a class prediction, and the prediction with the most votes becomes our model's final prediction.By using uncorrelated trees who give individual outputs, the Random Forest model generates a prediction by the committee that is more accurate than that of any individual tree.

                            </p>
                           <br>
                           <p>Confusion Matrix for Random Forests:</p>
                          <img src = "static/assets/img/randomforests.png" width = "400" height = "400" >
                           <br>
                          <br>
                          <br>
                           <ul class="list-group">
                             <li class="list-group-item list-group-item-primary">Accuracy : 82%</li>
                             <li class="list-group-item list-group-item-secondary">Precision : 0.87</li>
                             <li class="list-group-item list-group-item-success">Recall : 0.90</li>
                             <li class="list-group-item list-group-item-success">F1-Score : 0.88</li>
                           </ul>
                        </div>
                    </div>

                    <div class="col-6 col-sm-4">
                        <div class="team-member">
                            <h4>Neural Network </h4>
                            <img src="https://www.pngitem.com/pimgs/m/531-5314899_artificial-neural-network-png-transparent-png.png" height = "400" width = "400" alt="..." />
                            <p class="text-center" >A neural network is a method in artificial intelligence that teaches computers to process data in a way that is inspired by the human brain. It is a type of machine learning process, called deep learning, that uses interconnected nodes or neurons in a layered structure that resembles the human brain. Some projects that have used Neural Networking that are used in a day to day basis are weather forecasting,  credit scoring websites, and fraud detection systems.</p>
                            <br>
                            <p>Confusion Matrix for Neural Networks:</p>
                          <img src = "static/assets/img/neural_network.png" width = "400" height = "400" >
                            <br>
<br>
                          <br>
                            <ul class="list-group">
                              <li class="list-group-item list-group-item-primary">Accuracy : 84%</li>
                              <li class="list-group-item list-group-item-secondary">Precision : 0.87</li>
                              <li class="list-group-item list-group-item-success">Recall : 0.92</li>
                              <li class="list-group-item list-group-item-success">F1-Score :  0.90</li>
                            </ul>
                        </div>
                    </div>
                    <br>


                </div>
                <div class="row">
                    <div class="col-lg-8 mx-auto text-center"><p class="large text-muted"></p></div>
                </div>
            </div>
        </section>

        <!-- Contact-->
        <div class="py-5">
          <div class="container">
            <div class="row text-center">
              <div >
                  <a href="https://scikit-learn.org/stable/modules/model_evaluation.html"><h4 class="text-center " >Classification Metrics</h4></a>  <br>
                  <br>
              </div>
                <div class="col-md-6">
                  <h5>Confusion Matrix</h5>
                      <p>Confusion Matrix is a performance measurement for the machine learning classification problems where the output can be two or more classes. It is a table with combinations of predicted and actual values.</p>
                      <table class="table table-sm ">
                          <tbody>
                              <tr>
                                  <th > </th>
                                  <th >Predicted: 0 </th>
                                  <th >Predicted: 1</th>
                              </tr>
                              <tr>
                                  <td >Actual: 0</td>
                                  <td class="table-success">True Negative (<span class="bold">TN</span>)</td>
                                  <td class="table-danger">False Positive (<span class="bold">FP</span>)</td>
                              </tr>
                              <tr>
                                  <td >Actual: 1</td>
                                  <td class="table-danger">False Negative (<span class="bold">FN</span>)</td>
                                  <td class="table-success">True Positive (<span class="bold">TP</span>)</td>
                              </tr>
                          </tbody>
                      </table>

                      <ul  class="list-group">
                          <li ><strong>True Positives (TP)</strong>: The number of positive instances correctly classified as positive. E.g., predicting an email as spam when it actually is spam.</li>
                          <li ><strong>False Positives (FP)</strong>: The number of negative instances incorrectly classified as positive. E.g., predicting an email is spam when it actually is not spam.</li>
                          <li ><strong>True Negatives (TN)</strong>: The number of negative instances correctly classified as negative. E.g., predicting an email is not spam when it actually is not spam.</li>
                          <li ><strong>False Negatives (FN)</strong>: The number of positive instances incorrectly classified as negative. E.g., predicting an email is not spam when it actually is spam.</li>
                      </ul>
                      <br> <br>
                      <h5>Accuracy</h5>
                          <p>Accuracy is one metric for evaluating classification models. Informally, accuracy is the fraction of predictions our model got right.Accuracy is a good measure when there is class balance i.e. when both classes are almost equal or comparable. For imbalance classes other metrics such as precision , recall etc will give better perception on how model is performing on new data.</p>
                          <p class="text-center" style="color:red;font-size:20px;">Accuracy = \(\frac {TP+TN}{TP+TN+FP+FN}\)</p>

                      <br><br>
                </div>
                <div class="col-md-6">
                  <h5>Recall / Sensitivity </h5>
                      <p>Recall explains how many of the actual positive cases we were able to predict correctly with our model. It is a useful metric in cases where False Negative is of higher concern than False Positive. It is important in medical cases where it does not matter whether we raise a false alarm but the actual positive cases should not go undetected!</p>
                      <p class="text-center" style="color:red;font-size:20px;">Recall = \(\frac {TP}{TP+FN}\)</p> <br>
                  <h5>Precision</h5>
                      <p>Precision explains how many of the correctly predicted cases actually turned out to be positive. Precision is useful in the cases where False Positive is a higher concern than False Negatives. The importance of Precision is in music or video recommendation systems, e-commerce websites, etc. where wrong results could lead to customer churn and this could be harmful to the business.</p>
                      <p class="text-center" style="color:red;font-size:20px;">Precision = \(\frac {TP}{TP+FP}\)</p> <br>
                  <h5>F1 Score</h5>
                      <p>The F1-score (also sometimes called the F-Measure) is a single performance metric that takes both precision and recall into account. It's calculated by taking the harmonic mean of the two metrics.Only when both precision and recall have good performance will the F1-score be high.</p>
                      <p class="text-center" style="color:red;font-size:20px;">F1 Score = \(\frac {2 . Precision . Recall}{Precision + Recall}\)</p> <br>
                </div>
            </div>
          </div>
        </div>


<section class="page-section bg-dark text-white" id="Prediction">
  
 <div class="container">
     <div class="text-center">
         <h2 class="section-heading text-uppercase">Predict Salary</h2>
     </div>
     <!-- Make user predictions -->
     <form action="" method="post">
         <div class="row">
             <!-- age -->
             <div class="input-field col s4">
                 <label for="age"><b>Age</b></label>
                 <br>
                 <input id="age" name="age" type="number" placeholder="Enter age" class="validate" >
             </div>
             <!-- Workclass -->
             <div class="input-field col s4">
                 <label for="workclass"><b>Workclass</b></label>
                 <br>
                 <input list="workclass" name="workclass" type="text" placeholder="Enter workclass" class="validate" >
                   <datalist id="workclass">
                     <option value="State-gov"></option>
                     <option value="Self-emp-not-inc"></option>
                     <option value="Private"></option>
                     <option value="Federal-gov"></option>
                     <option value="Other-service"></option>
                     <option value="Local-gov"></option>
                     <option value="Self-emp-inc"></option>
                     <option value="Without-pay"></option>
                     <option value="Never-worked"></option>
               </datalist>

             </div>
             <!-- education -->
             <div class="input-field col s4">
                 <label for="education"><b>No of Years of Education</b></label>
                 <br>
                 <input id="education" name="education" type="text" placeholder="Education" class="validate" >
             </div>
             <!-- Living Area -->
             <div class="input-field col s4">
                 <label for="martial"><b>Martial Status</b></label>
                 <br>
                 <input list="martial" name="martial" type="text" placeholder="Enter workclass" class="validate" >
                   <datalist id="martial">
                     <option value="Never-married"></option>
                     <option value="Married-civ-spouse"></option>
                     <option value="Divorced"></option>
                     <option value="Married-spouse-absent"></option>
                     <option value="Separated"></option>
                     <option value="Married-AF-spouse"></option>
                     <option value="Widowed"></option>
               </datalist>

             </div>
             <div class="w-100"></div>
             <br>

             <!-- Home Type -->
             <div class="input-field col s4">
                 <label for="occupation"><b>occupation</b></label>
                 <br>
                 <input list="occupation" name="occupation" type="text" placeholder="Enter occupation" class="validate" >
                   <datalist id="occupation">
                     <option value="Adm-clerical"></option>
                     <option value="Exec-managerial"></option>
                     <option value="Handlers-cleaners"></option>
                     <option value="Prof-specialty"></option>
                     <option value="Other-service"></option>
                     <option value="Sales"></option>
                     <option value="Craft-repair"></option>
                     <option value="Transport-moving"></option>
                     <option value="Farming-fishing"></option>
                     <option value="Machine-op-inspct"></option>
                      <option value="Tech-support"></option>
                     <option value="Protective-serv"></option>
                     <option value="Armed-Forces"></option>
                     <option value="Priv-house-serv"></option>
 
                  </datalist>

             </div>
           
             <div class="input-field col s4">
                 <label for="Relationship"><b>Relationship</b></label>
                 <br>
                 <input list="Relationship" name="Relationship" type="text" placeholder="Enter Relationship" class="validate" >
               <datalist id="Relationship">
                     <option value="Not-in-family">
                     </option><option value="Husband">
                     </option><option value="Wife">
                     </option><option value="Unmarried">
                     </option><option value="Other-relative">
                   </option></datalist>

             </div>
           
             <div class="input-field col s4">
                 <label for="Race"><b>Race</b></label>
                 <br>
                 <input list="Race" name="Race" placeholder="Enter race" >
                   <datalist id="Race">
                     <option value="White"></option>
                     <option value="Black"></option>                                                         <option value="Asian-Pac-Islande"></option>
                     <option value="Amer-Indian-Eskimo"></option>
                     <option value="Other"></option>
                   </datalist>
             </div>

             <div class="input-field col s4">
                 <label for="Gender"><b>Gender</b></label>
                 <br>
                 <input list="Gender" name="Gender" placeholder="Male or Female">
                   <datalist id="Gender">
                     <option value="Male">
                     </option><option value="Female">
                   </option></datalist>
             </div>

             <div class="w-100"></div>
             <br>
             
              <div class="input-field col s4">
                <p>
                  
                </p>
             </div>

             <div class="input-field col s4">
                 <label for="hours_per_week"><b>hours_per_week</b></label>
                 <br>
                 <input id="hours_per_week" name="hours_per_week" type="number" placeholder="No of hours per week" class="validate" >
             </div>



             <div class="input-field col s4">
                 <label for="Native_country"><b>Native Country</b></label>
                 <br>
                 <input list="Native_country" name="Native_country" placeholder="Country" >
                   <datalist id="Native_country">
                     <option value="United-states">
                     </option><option value="others">
                   </option></datalist>
             </div>
           
           
           <div class="input-field col s4">
                <p>
                  
                </p>
             </div>


         </div>
         <br>
       
         
    <div class="container">
     <div class="text-center">
             <button type="submit" class="btn btn-primary btn-l text-uppercase">Predict Salary</button>
             <br>
            <table class="table table-dark">
                {{df_html | safe}}
            </table>
            <p><h5>{{values}}</h5></p>
         
       </div>
      </div>



     </form>
 </div>
</section
  
  
<!--Conclusion-->
          <div class="container">
            <br>
          <div class="row text-center">
            <div >
                <h2 class="text-center" >Conclusion</h2>  <br>

                <p>After analyzing our data, we found that out of all of the classification models, Neural Networks worked the best for predicting salary. Our Logistic Regression and Random Forest models both had an accuracy of around 82%, while our Neural Network model had an accuracy of around 84%, making it the best of the three that we used. From the analysis we can conclude that there is a relation between level of education, age, race etc.. and income received by a person.
 </p>
                <br>
<!--             </div>
            
            <div>
               <h2 class="text-center" >Potential Improvements</h2>  <br>
            <p> </p>
              
              
            </div>
          </div>
        </div> -->
         
         
        <!-- Team-->
        <section class="page-section bg-light" id="team">
            <div class="container">
                <div class="text-center">
                    <h2 class="section-heading text-uppercase">Team</h2>
                    <h3 class="section-subheading text-muted">Aka the Electric Zombies</h3>
                </div>
                <div class="row">
                      <div class="col-lg-4">
                        <div class="team-member">
                            <img class="mx-auto rounded-circle" src="https://c.tenor.com/0gR71awIDG0AAAAd/he-man.gif" alt="..." />
                            <h4>Aaron Chang</h4>
                            <p class="text-muted">Experimenting until I find what works</p>

                        </div>
                    </div>

                    <div class="col-lg-4">
                        <div class="team-member">
                            <img class="mx-auto rounded-circle" src="https://i.gifer.com/D8Jw.gif" alt="..." />
                            <h4>Adam Ellington</h4>
                            <p class="text-muted">The best way to predict the future is to create it</p>

                        </div>
                    </div>
                    <div class="col-lg-4">
                        <div class="team-member">
                            <img class="mx-auto rounded-circle" src="https://media1.giphy.com/media/dBrYsgU0BnSGAzSEdt/giphy.gif" alt="..." />
                            <h4>Amelia Lipcsei</h4>
                            <p class="text-muted">Winner of the worst wifi award</p>

                        </div>
                    </div>
                </div>
            </div>
            <div class="container">
                <div class="row">
                    <div class="col-lg-4">
                        <div class="team-member">
                            <img class="mx-auto rounded-circle" src="https://c.tenor.com/MGTw6hJMklUAAAAM/card-magic-world-xm.gif" alt="..." />
                            <h4>Joshua Broyer</h4>
                            <p class="text-muted">Model Analyst || Backend Developer || Musician</p>

                        </div>
                    </div>
                      <div class="col-lg-4">
                        <div class="team-member">
                            <img class="mx-auto rounded-circle" src="http://24.media.tumblr.com/bcb641dfcd087fdde2b877c3d9f31138/tumblr_mhyqumI6521rl9mjvo1_500.gif" alt="..." />
                            <h4>Karen Gerges</h4>
                            <p class="text-muted">I did my best and that’s all that matters</p>

                        </div>
                    </div>
                      <div class="col-lg-4">
                        <div class="team-member">
                            <img class="mx-auto rounded-circle" src="https://i.pinimg.com/originals/e5/10/aa/e510aae2296e8b01b0db27a24255b156.gif" alt="..." />
                            <h4>Shellene Redhorse</h4>
                            <p class="text-muted">Team Member</p>

                        </div>
                    </div>
                </div>
            </div>
          
            <div class="team-member">
              <img class="mx-auto rounded-circle" src="https://c.tenor.com/T2CaihGMlnEAAAAM/liverpool-klopp.gif" alt="..." />
              <h4>Vishnu Nelapati</h4>
              <p class="text">Instructor</p>
              <p class="text-muted">You Will Never Walk Alone</p>

          </div>
        </section>

        <!-- Footer-->
        <footer class="footer py-4">
            <div class="container">
                <div class="row align-items-center">
                    <div align = "center">Copyright &copy; AI Camp 2022</div>

                </div>
            </div>
        </footer>


        <!-- Portfolio Modals-->
        <!-- Portfolio item 1 modal popup-->
        <div class="portfolio-modal modal fade" id="portfolioModal1" tabindex="-1" role="dialog" aria-hidden="true">
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="close-modal" data-bs-dismiss="modal"><img src="static/assets/img/close-icon.svg" alt="Close modal" /></div>
                    <div class="container">
                        <div class="row justify-content-center">
                            <div class="col-lg-12">
                                <div class="modal-body">
                                    <!-- Project details-->
                                    <h2 class="text-uppercase">Gender Distribution</h2>
                                    <p class="item-intro text-muted">Is there any correlation of salary between genders?
                                      <object data="static/assets/img/portfolio/plot1.html"
                                          width="1020"
                                          height="520"
                                          type="text/html">
                                      </object>
                                    <p> According to the data there is some correlation to males on average getting paid more than women.
There are certain variables such as common jobs for each gender and certain skills or prioritize less than others like communication." </p>

                                    <button class="btn btn-primary btn-xl text-uppercase" data-bs-dismiss="modal" type="button">
                                        <i class="fas fa-xmark me-1"></i>
                                        Close
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <!-- Portfolio item 2 modal popup-->
        <div class="portfolio-modal modal fade" id="portfolioModal2" tabindex="-1" role="dialog" aria-hidden="true">
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="close-modal" data-bs-dismiss="modal"><img src="static/assets/img/close-icon.svg" alt="Close modal" /></div>
                    <div class="container">
                        <div class="row justify-content-center">
                            <div class="col-lg-12">
                                <div class="modal-body">
                                    <!-- Project details-->
                                    <h2 class="text-uppercase">Gender Pay Gap </h2>
                                    <p class="item-intro text-muted">Does more years of education determine higher salary?</p>
                                    <object data="static/assets/img/portfolio/plot2.html"
                                        width="1020"
                                        height="520"
                                        type="text/html">
                                    </object>
                                    <p> Typically more years of education is intertwined with an higher salary. This may be because certain occupations
require more knowledge which in turns gives more salary because the position is more valuable.
There is also a trend that women have to have more education years in order to make the same as their gender counterparts .</p>

                                    <button class="btn btn-primary btn-xl text-uppercase" data-bs-dismiss="modal" type="button">
                                        <i class="fas fa-xmark me-1"></i>
                                        Close
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <!-- Portfolio item 3 modal popup-->
        <div class="portfolio-modal modal fade" id="portfolioModal3" tabindex="-1" role="dialog" aria-hidden="true">
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="close-modal" data-bs-dismiss="modal"><img src="static/assets/img/close-icon.svg" alt="Close modal" /></div>
                    <div class="container">
                        <div class="row justify-content-center">
                            <div class="col-lg-12">
                                <div class="modal-body">
                                    <!-- Project details-->
                                    <h2 class="text-uppercase">Race vs Salary</h2>
                                    <p class="item-intro text-muted">Is there any correlation between higher pay and race?</p>

                                    <object data="static/assets/img/portfolio/plot3.html"
                                        width="1020"
                                        height="520"
                                        type="text/html">
                                    </object>
                                    <p>Despite Asian-pacific islanders having more education years on average they are still getting paid less than their race counter parts.
This may be due to Asian-pacific islanders having the largest sample size and having the largest income inequality which results in them having the smaller pay. In conclusion Asian - Pacific Islander need to have more years of education to earn more salary compared to other races.!</p>

                                    <button class="btn btn-primary btn-xl text-uppercase" data-bs-dismiss="modal" type="button">
                                        <i class="fas fa-xmark me-1"></i>
                                        Close
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <!-- Portfolio item 4 modal popup-->
        <div class="portfolio-modal modal fade" id="portfolioModal4" tabindex="-1" role="dialog" aria-hidden="true">
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="close-modal" data-bs-dismiss="modal"><img src="static/assets/img/close-icon.svg" alt="Close modal" /></div>
                    <div class="container">
                        <div class="row justify-content-center">
                            <div class="col-lg-12">
                                <div class="modal-body">
                                    <!-- Project details-->
                                    <h2 class="text-uppercase">Age Distribution vs Salary</h2>
                                    <p class="item-intro text-muted">Does age correlate to higher salary?</p>
                                    <object data="static/assets/img/portfolio/plot4.html"
                                        width="1020"
                                        height="520"
                                        type="text/html">
                                    </object>
                                    <p>On average the older an individual is, the more he/she will make. Experience is a huge part of the work field and with more experience shows a greater mastery over that skill. Which in turn makes them more valuable to the company and will receive a higher salary then to those who are just starting.!</p>

                                    <button class="btn btn-primary btn-xl text-uppercase" data-bs-dismiss="modal" type="button">
                                        <i class="fas fa-xmark me-1"></i>
                                        Close
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <!-- Portfolio item 5 modal popup-->
        <div class="portfolio-modal modal fade" id="portfolioModal5" tabindex="-1" role="dialog" aria-hidden="true">
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="close-modal" data-bs-dismiss="modal"><img src="static/assets/img/close-icon.svg" alt="Close modal" /></div>
                    <div class="container">
                        <div class="row justify-content-center">
                            <div class="col-lg-12">
                                <div class="modal-body">
                                    <!-- Project details-->
                                    <h2 class="text-uppercase">Highest Level of education vs Salary</h2>
                                    <p class="item-intro text-muted">Does higher education determine higher salary?.</p>
                                    <object data="static/assets/img/portfolio/plot5.html"
                                        width="1120"
                                        height="620"
                                        type="text/html">
                                    </object>
                                    <p>The higher education an individual receives it increases the likelihood of making over 50k. It is very unlikely that a person who has obtained his master will make under 50k, as shown on graph there is a 3 percent to make under while 12 percent make over. Nothing is guaranteed to work out but an higher education increases the probability for it to succeed!</p>

                                    <button class="btn btn-primary btn-xl text-uppercase" data-bs-dismiss="modal" type="button">
                                        <i class="fas fa-xmark me-1"></i>
                                        Close
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <!-- Portfolio item 6 modal popup-->
        <div class="portfolio-modal modal fade" id="portfolioModal6" tabindex="-1" role="dialog" aria-hidden="true">
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="close-modal" data-bs-dismiss="modal"><img src="static/assets/img/close-icon.svg" alt="Close modal" /></div>
                    <div class="container">
                        <div class="row justify-content-center">
                            <div class="col-lg-8">
                                <div class="modal-body">
                                    <!-- Project details-->
                                    <h2 class="text-uppercase">Correlation HeatMap</h2>
                                    <p class="item-intro text-muted">Identifying the columns that are correlating with salary.</p>
                                    <object data="static/assets/img/portfolio/plot6.html"
                                        width="620"
                                        height="620"
                                        type="text/html">
                                    </object>
                                    <p> As we can observe salary has positive correlation with age , education number and hours_per_week worked. Education number has more influence on salary compared to other features.</p>

                                    <button class="btn btn-primary btn-xl text-uppercase" data-bs-dismiss="modal" type="button">
                                        <i class="fas fa-xmark me-1"></i>
                                        Close
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <!-- Bootstrap core JS-->
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <!-- Core theme JS-->
        <script src="static/js/scripts.js"></script>
        <!-- * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *-->
        <!-- * *                               SB Forms JS                               * *-->
        <!-- * * Activate your form at https://startbootstrap.com/solution/contact-forms * *-->
        <!-- * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *-->
        <script src="https://cdn.startbootstrap.com/sb-forms-latest.js"></script>
    </body>
</html>
