import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
df=pd.read_csv('assigned_dataset.csv')
df
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
print(df.columns)
print(df.isnull().sum())
print(df.info())

#1
df['ParentMaritalStatus'].value_counts()
df['ParentMaritalStatus'] = df['ParentMaritalStatus'].str.lower().str.strip()
df['ParentMaritalStatus'].replace(['married','Married'],'married',inplace=True)
df['ParentMaritalStatus'].replace(['widow','Widow'],'widow',inplace=True)
df['ParentMaritalStatus'].replace(['seperated','Seperated'],'Seperated',inplace=True)
#df['ParentMaritalStatus'].replace(['0'], 'unknown', inplace=True) 

print(df['ParentMaritalStatus'].value_counts())

#2
df['Father education'].value_counts()
df['Father education'] = df['Father education'].str.lower().str.strip()
df['Father education'].replace(['graduate', 'graduation', 'under graduate'], 'graduate', inplace=True)
df['Father education'].replace(['post graduate', 'post-graduate', 'post graduate ', 'post graduation','post - graduate'], 'post graduate', inplace=True)
df['Father education'].replace(['diploma', 'diploma '], 'diploma', inplace=True)
df['Father education'].replace(['high school', 'high school ', 'high - school','higher secondary'], 'high school', inplace=True)
#df['Father education'].replace(['0'], 'unknown', inplace=True)  # Assuming 0 represents unknown

print(df['Father education'].value_counts())

#3
df['Mother education'].value_counts()
df['Mother education'] = df['Mother education'].str.lower().str.strip()

df['Mother education'].replace(['graduate', 'graduation', 'under graduate'], 'graduate', inplace=True)
df['Mother education'].replace(['post graduate', 'post-graduate', 'post graduate ', 'post graduation'], 'post graduate', inplace=True)
df['Mother education'].replace(['diploma', 'diploma '], 'diploma', inplace=True)
df['Mother education'].replace(['high school', 'high school ', 'high - school', 'highschool','higher secondary'], 'high school', inplace=True)
#df['Mother education'].replace(['0'], 'unknown', inplace=True) 
df['Mother education'].replace(['post-graduate', 'post - graduate', 'post graduate '], 'post graduate', inplace=True)
df['Mother education'].replace(['high-school', 'highschool', 'high  school', 'high'], 'high school', inplace=True)
print(df['Mother education'].value_counts())

#4
df['Father Occupation'].value_counts()
df['Father Occupation'] = df['Father Occupation'].str.lower().str.strip()

df['Father Occupation'].replace(['private', 'Private '], 'private', inplace=True)
df['Father Occupation'].replace(['government', 'government ', 'government           '], 'government', inplace=True)
df['Father Occupation'].replace(['self employed', 'self - employeed', 'self- employed', 'self-employeed', 'self employed', 'self-employed', 'self employed ', 'self-employed ', 'self employed        ', 'self-employed         '], 'self-employed', inplace=True)
df['Father Occupation'].replace(['self employed', 'self - employeed', 'self- employed', 'self-employeed','----','service'], 'self-employed', inplace=True)
#df['Father Occupation'].replace(['0'], 'unknown', inplace=True)

print(df['Father Occupation'].value_counts())

#5
df['Mother Occupation'].value_counts()
df['Mother Occupation'] = df['Mother Occupation'].str.lower().str.strip()

df['Mother Occupation'].replace(['private', 'private ', 'private              '], 'private', inplace=True)
df['Mother Occupation'].replace(['self employed','self - employeed','self- employed','self-employed','self- employeed', 'self employed ', 'self-employed ', 'self employed        ', 'self-employed         ', 'self employed         '], 'self-employed', inplace=True)
df['Mother Occupation'].replace(['government', 'government ', 'government           '], 'government', inplace=True)
df['Mother Occupation'].replace(['self employeed', 'self employeed       ', 'self-employeed','service'], 'self-employed', inplace=True)
df['Mother Occupation'].replace(['high school', 'house wife', 'home maker'], 'House wife', inplace=True)
df['Mother Occupation'].replace(['business'], 'Business', inplace=True)

#df['Mother Occupation'].replace(['0'], 'unknown', inplace=True)  # Assuming 0 represents unknown

print(df['Mother Occupation'].value_counts())

#6
df['Socially Skills'].value_counts()
df['Socially Skills'] = df['Socially Skills'].str.lower().str.strip()

df['Socially Skills'].replace(['active','Active'], 'active', inplace=True)
df['Socially Skills'].replace(['inactive'], 'inactive', inplace=True)
df['Socially Skills'].replace(['moderately active', 'Moderately Active', 'moderately active', 'moderately-active', 'moderately actt'], 'moderately active', inplace=True)

print(df['Socially Skills'].value_counts())

df['Socially Skills'] = df['Socially Skills'].str.strip()
Social_skills_counts=df['Socially Skills'].value_counts()[:3]
plt.pie(Social_skills_counts,labels=Social_skills_counts.index,autopct=lambda p: '{:}%'.format(p) if p > 0 else '',startangle=90)
plt.title('Top 3 Socially skills')
plt.show()

#7
df['Teacher Interaction'].value_counts()
df['Teacher Interaction'] = df['Teacher Interaction'].str.lower().str.strip()

df['Teacher Interaction'].replace(['good','Good'], 'good', inplace=True)
df['Teacher Interaction'].replace(['nil', 'nill','Nil'], 'nil', inplace=True)
df['Teacher Interaction'].replace(['moderate','Moderate'], 'moderate', inplace=True)

print(df['Teacher Interaction'].value_counts())

#8
df['Cognitive Development'].value_counts()
df['Cognitive Development'] = df['Cognitive Development'].str.lower().str.strip()

df['Cognitive Development'].replace(['good', 'good ', 'GOOD','average', 'Average ', 'AVERAGE'], 'good', inplace=True)
df['Cognitive Development'].replace(['low', 'low ','LOW','gps'], 'low', inplace=True)
df['Cognitive Development'].replace(['excellent', 'Excellent ', 'EXCELLENT'], 'excellent', inplace=True)

print(df['Cognitive Development'].value_counts())

#9
df['Technology Influence'].value_counts()
df['Technology Influence'] = df['Technology Influence'].str.lower().str.strip()

df['Technology Influence'].replace(['moderate', 'moderate ', 'modeerate','Modeerate', 'moderate        ', '. moderate'], 'moderate', inplace=True)
df['Technology Influence'].replace(['addicted', 'Addicted '], 'addicted', inplace=True)
df['Technology Influence'].replace(['no usage', 'no usage ', 'No Usage', 'No Usage ', 'No usage', 'No usage '], 'no usage', inplace=True)

print(df['Technology Influence'].value_counts())

#10
df['Social Media Influence'].value_counts()
df['Social Media Influence'] = df['Social Media Influence'].str.lower().str.strip()

df['Social Media Influence'].replace(['inactive', 'inactive ', 'Inactive'], 'inactive', inplace=True)
df['Social Media Influence'].replace(['active', ' Active','moderately active', 'moderately-active', ' Moderately Active', 'moderately active '], 'moderately active', inplace=True)
df['Social Media Influence'].replace(['over active', 'over-active', ' Over active'], 'over active', inplace=True)
#df['Social Media Influence'].replace(['0'], 'unknown', inplace=True)  # Assuming 0 represents unknown

print(df['Social Media Influence'].value_counts())

#11
df['Involved in Extra-Curricular'].value_counts()
df['Involved in Extra-Curricular'] = df['Involved in Extra-Curricular'].str.lower().str.strip()

df['Involved in Extra-Curricular'].replace(['regular','regularly', 'regularly ', 'Regularly'], 'regularly', inplace=True)
df['Involved in Extra-Curricular'].replace(['sometimes', 'Sometimes', 'sometime'], 'sometimes', inplace=True)
df['Involved in Extra-Curricular'].replace(['never', 'never ', 'Never', 'neve'], 'never', inplace=True)
#df['Involved in Extra-Curricular'].replace(['0'], 'unknown', inplace=True)  # Assuming 0 represents unknown

print(df['Involved in Extra-Curricular'].value_counts())

#12
df['Self-Esteem'].value_counts()
df['Self-Esteem'] = df['Self-Esteem'].str.lower().str.strip()

# Replace similar entries
df['Self-Esteem'].replace(['low', 'low confident', 'low- confident', 'lo'], 'low', inplace=True)
df['Self-Esteem'].replace(['confident', 'over confident', 'over- confide'], 'confident', inplace=True)
df['Self-Esteem'].replace(['over- confident', 'over-confident', 'over- confide', 'over- confident'], 'over confident', inplace=True)
df['Self-Esteem'].replace(['over- confident', 'over-confident', 'over- confide', 'over- confident'], 'over confident', inplace=True)

# Display the cleaned and standardized counts
print(df['Self-Esteem'].value_counts())

#13
df['Attentivity in Class'].value_counts()

df['Attentivity in Class'] = df['Attentivity in Class'].str.lower().str.strip()

# Replace similar entries
df['Attentivity in Class'].replace(['good','Good'], 'good', inplace=True)
df['Attentivity in Class'].replace(['nil','very poor'], 'nil', inplace=True)
df['Attentivity in Class'].replace(['moderate'], 'moderate', inplace=True)
df['Attentivity in Class'].replace(['ggod'], 'good', inplace=True)

# Display the cleaned and standardized counts
print(df['Attentivity in Class'].value_counts())

#14
df['Attendance'].value_counts()
df['Attendance'] = df['Attendance'].str.strip()
df['Attendance'].replace(['90','0.9','0.91','0.92','0.93','0.94','0.95','0.96','0.97','0.98','0.99','90%','91%','92%','93%','94%','95%','96%','97%','98%','99%','100%'], 'Excellent', inplace=True)
df['Attendance'].replace(['0.88','70%','0.71','0.72','0.73','0.74','0.75','0.76','0.77','0.78','0.79','0.8','0.81','0.82','0.83','0.84','0.85','0.86','0.87','0.89','81%','82%','83%','84%','85%','86%','87%','88%','89%','70%','71%','72%','73%','74%','75%','76%','77%','78%','79%','80%'], 'Good',inplace=True)
df['Attendance'].replace(['40%','48%','35%','30%','0.6','0.61','0.63','61%','62%','63%','64%','65%','66%','67%','68%','69%','45%','50%','51%','52%','53%','54%','55%','56%','57%','58%','59%','60%'], 'Low',inplace=True)
print(df['Attendance'].value_counts())

#15
df['Homework completion'].value_counts()

df['Homework completion'] = df['Homework completion'].str.lower().str.strip()

df['Homework completion'].replace(['always','Always','alwyas','Alwyas'], 'always', inplace=True)
df['Homework completion'].replace(['never','Never','Very poor','very poor'], 'never', inplace=True)
df['Homework completion'].replace(['sometimes','Sometimes'], 'sometimes', inplace=True)

print(df['Homework completion'].value_counts())

#16
df['PracticeSport'].value_counts()
df['PracticeSport'] =df['PracticeSport'].str.lower().str.strip()

# Replace similar entries
df['PracticeSport'].replace(['regular', 'regularly'], 'regularly', inplace=True)
df['PracticeSport'].replace(['never','very poor','never.'], 'never', inplace=True)
df['PracticeSport'].replace(['sometimes', 'Regularly'], 'sometimes', inplace=True)

# Display the cleaned and standardized counts
print(df['PracticeSport'].value_counts())

#17
df['Age'].value_counts()
sns.countplot(x='Age', data=df)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Value Counts of Age')
plt.show()

#18
df['Gender'].value_counts()

df['Gender'] = df['Gender'].str.lower().str.strip()
df['Gender'].replace(['male', 'm'], 'Male', inplace=True)
df['Gender'].replace(['female', 'f'], 'Female', inplace=True)

print(df.Gender.value_counts())

#19
df['Class'].value_counts()

#20
df.Performance.value_counts()
# Standardize the values in the 'Performance' column
df['Performance'] = df['Performance'].str.lower().str.strip()

# Replace similar values with a common representation
df['Performance'].replace(['Average'], 'Average', inplace=True)
df['Performance'].replace(['Excellent'], 'Excellent', inplace=True)
df['Performance'].replace(['Good'], 'Good', inplace=True)
df['Performance'].replace(['Poor'], 'Poor', inplace=True)

print(df['Performance'].value_counts())
sns.countplot(x='Performance', data=df)
plt.xlabel('Performance')
plt.ylabel('Count')
plt.title('Value Counts of Performance')
plt.show()

#21
df['Behavioral Patterns'].value_counts()
# Standardize the values in the 'Behavioral Patterns' column
# Standardize the values in the 'Behavioral Patterns' column
df['Behavioral Patterns'] = df['Behavioral Patterns'].str.lower().str.strip()

# Replace similar values with a common representation (e.g., 'calm' for all variations)
df['Behavioral Patterns'].replace(['caml', 'clam','calm','Calm'], 'calm', inplace=True)
df['Behavioral Patterns'].replace(['active','focussed','Focussed', 'foused','Focused','focused'], 'focused', inplace=True)
df['Behavioral Patterns'].replace(['hyper active', 'hyperactive ',' hyperactive'], 'hyperactive', inplace=True)
df['Behavioral Patterns'].replace([' Aggressive','aggressive','disturbed','diaturbed','aggressive'], 'aggressive', inplace=True)
df['Behavioral Patterns'].replace(['sleepy'],'sleepy',inplace=True)
print(df['Behavioral Patterns'].value_counts())

sns.countplot(x='Behavioral Patterns', data=df)
plt.xlabel('Behavioral Patterns')
plt.ylabel('Count')
plt.title('Value Counts of Behavioural Patterns')
plt.show()

#22
df['Health Issues'].value_counts()
df['Health Issues'] = df['Health Issues'].str.lower().str.strip()

df['Health Issues'].replace(['n', 'n '], 'no', inplace=True)
df['Health Issues'].replace(['y', 'y '], 'yes', inplace=True)
df['Health Issues'].replace(['0'], 'unknown', inplace=True)  # Assuming 0 represents unknown

print(df['Health Issues'].value_counts())
sns.countplot(x='Health Issues', data=df)
plt.xlabel('Health Issues')
plt.ylabel('Count')
plt.title('Value Counts of Heath Issues')
plt.show()

#23
df['Academic Score'].value_counts()
df['Academic Score'] = df['Academic Score'].str.strip()
df['Academic Score'].replace(['93.10%','90','0.9','0.91','0.92','0.93','0.94','0.95','0.96','0.97','0.98','0.99','90%','91%','92%','93%','94%','95%','96%','97%','98%','99%','100%'], 'Excellent', inplace=True)
df['Academic Score'].replace(['0.62','0.67','75','89.50%','0.7','0.65','0.69','8502%','0.6','0.61','0.63','61%','62%','63%','64%','65%','66%','67%','68%','69%','0.88','70%','0.71','0.72','0.73','0.74','0.75','0.76','0.77','0.78','0.79','0.8','0.81','0.82','0.83','0.84','0.85','0.86','0.87','0.89','81%','82%','83%','84%','85%','86%','87%','88%','89%','71%','72%','73%','74%','75%','76%','77%','78%','79%','80%'], 'Good',inplace=True)
df['Academic Score'].replace(['59.60%','0.46','0.43','0.64','0.68','0.47','0.52','0.5','46.50%','59.60','0.48','43.50%','0.45','45.60%','0.4','0.57','0.58','0.55','0.59','40%','41%','42%','43%','44%','46%','47%','49%','48%','45%','50%','51%','52%','53%','54%','55%','56%','57%','58%','59%','60%'], 'Low',inplace=True)
df['Academic Score'].replace(['15%','0.37','0.32','0.3','4%','6%','0.39','11%','18%','16%','3%','36%','10%','35%','30%','31%','32%','33%','34%','35%',',36%','37%','38%','39%','20%','21%','22%','23%','24%','25%','26%','27%','28%','29%','30%'],'very poor',inplace=True)
print(df['Academic Score'].value_counts())

sns.countplot(x='Academic Score', data=df)
plt.xlabel('Academic Score')
plt.ylabel('Count')
plt.title('Value Counts of Academic Score')
plt.show()

#24
df['Social Media Influence'].value_counts()
df['Social Media Influence'] = df['Social Media Influence'].str.strip()
social_media_counts = df['Social Media Influence'].value_counts()[:3]

# Plotting a pie chart for the top three values with customized autopct formatting
plt.pie(social_media_counts, labels=social_media_counts.index, autopct=lambda p: '{:.1f}%'.format(p) if p > 0 else '', startangle=90)
plt.title('Top Social Media Influences')
plt.show()

df.isnull().sum()
df.info()
# Get column names where data type is 'object'
object_columns = df.select_dtypes(include=['object']).columns

# Fill null values in object columns with the most frequent value
for col in object_columns:
    most_frequent_value = df[col].mode()[0]
    df[col].fillna(most_frequent_value, inplace=True)
most_frequent_value = df['Size of Family'].mode()[0]
df['Size of Family'].fillna(most_frequent_value, inplace=True)

#25
print(df['Size of Family'].value_counts())

#26
df['Class'].value_counts()

# Replace 'nur' and 'N' with 11, and 'KG' with 0
df['Class'].replace({'nur': 11, 'N': 11, 'KG': 0}, inplace=True)

print(df['Class'].value_counts())

df.isnull().sum()

print(df.head())
df.info()
df.columns
from sklearn.preprocessing import LabelEncoder

# Create a label encoder object
label_encoder = LabelEncoder()

# Iterate over each column in the DataFrame
for column in df.columns:
    # Check if the column data type is 'object' and if the column is not one of the excluded columns
    if df[column].dtype == 'object' and column not in ['Age', 'Class', 'Size of Family']:
        # Apply label encoding to the column
        df[column] = label_encoder.fit_transform(df[column])

# Now your DataFrame 'df' contains label encoded values for all object columns except the excluded ones

print(df.head())

df.to_csv("Cleaned_dataset.csv")
X = df.drop(columns=['Performance'])

# y will contain only the 'Performance' column
Y = df['Performance']

print(Y.value_counts())

print(X.head())
X.to_csv("input.csv")
Y.to_csv("Out.csv")
# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train


#decision tree 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Initialize and fit your model
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, y_train)
# Evaluate your CatBoost model
y_pred = model_dt.predict(X_test)

Decision_tree_accuracy= accuracy_score(y_test, y_pred)


#Adaboost accuracy
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
model_ada = AdaBoostClassifier()

# Fit the model
model_ada.fit(X_train, y_train)
# Evaluate your CatBoost model
y_pred = model_ada.predict(X_test)

Adaboost_accuracy = accuracy_score(y_test, y_pred)


#catboost
from catboost import CatBoostClassifier
from sklearn.metrics import precision_score
model_cat = CatBoostClassifier()
model_cat.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)  # Set verbose to True for training logs

# Evaluate your CatBoost model
y_pred = model_cat.predict(X_test)

Catboost_accuracy = accuracy_score(y_test, y_pred)

#dumping into a pickle file
pickle.dump(model_cat,open('model.pkl','wb'))


# Evaluate models
models = [model_dt, model_ada, model_cat]
model_names = ['Decision Tree', 'AdaBoost', 'CatBoost']
metrics = ['Train Accuracy', 'Test Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
results = []

for model, name in zip(models, model_names):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)    
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    precision = precision_score(y_test, test_pred, average='weighted')
    recall = recall_score(y_test, test_pred,average='weighted')
    f1 = f1_score(y_test, test_pred, average='weighted')
    auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
    
    results.append([train_acc*100, test_acc*100, precision*100, recall*100, f1*100, auc*100])


# Create DataFrame
df = pd.DataFrame(results, columns=metrics, index=model_names)

# Plotting
plt.subplots(figsize=(10, 6))
# Plotting train accuracy
plt.bar(model_names, df['Train Accuracy'], color=['blue', 'orange', 'green'])
plt.title('Train Accuracy')
plt.show()


# Plotting
plt.subplots(figsize=(10, 6))
# Plotting test accuracy
plt.bar(model_names, df['Test Accuracy'], color=['blue', 'orange', 'green'])
plt.title('Test Accuracy')
plt.show()


# Plotting
plt.subplots(figsize=(10, 6))
# Plotting precision
plt.bar(model_names, df['Precision'], color=['blue', 'orange', 'green'])
plt.title('Precision Score Accuracy')
plt.show()

# Plotting
plt.subplots(figsize=(10, 6))
# Plotting recall
plt.bar(model_names , df['Recall'], color=['blue', 'orange', 'green'])
plt.title('Recall Accuracy')
plt.show()

# Plotting
plt.subplots(figsize=(10, 6))
# Plotting f1score accuracy
plt.bar(model_names, df['F1 Score'], color=['blue', 'orange', 'green'])
plt.title('F1 Score Accuracy')
plt.show()

# Plotting
plt.subplots(figsize=(10, 6))
# Plotting auc accuracy
plt.bar(model_names, df['AUC'], color=['blue', 'orange', 'green'])
plt.title('AUC Accuracy')
plt.tight_layout()
plt.show()

acc=np.array([Decision_tree_accuracy*100,Adaboost_accuracy*100,Catboost_accuracy*100])

x=['Decision tree','Adaboost','Catboost']
plt.bar(x,acc,color=('red','blue','green'))
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.title('Comparison of algorithms')
plt.show()


df['Accuracy']=acc

print(df)


test_vector = model_cat.predict([[10,1,6,4,2,2,3,1,1,0,0,1,2,2,1,4,1,2,1,1,0,2,0]])
test_vector
if test_vector[0]==0:
    print("Average student")
elif test_vector[0]==1:
    print("Excellent student")
elif test_vector[0]==2:
    print("Good student")
elif test_vector[0]==3:
    print("Poor student")
    

    
# Get input from the user
#age = int(input("Enter age: "))
#gender = int(input("Enter gender (0 for female, 1 for male): "))
#class_size = int(input("Enter your standard: "))
#family_size = int(input("Enter size of family: "))
#father_education = int(input("Enter father's education level(0 for diploma, 1 for graduate, 2 for high school, 3 for post-graduate): "))
#mother_education = int(input("Enter mother's education level(0 for diploma, 1 for graduate, 2 for high school, 3 for post-graduate): "))
#father_occupation = int(input("Enter father's occupation(0 for business, 1 for government, 2 for private, 3 self-employeed): "))
#mother_occupation = int(input("Enter mother's occupation(0 for business, 1 for house wife, 2 for government, 3 for private, 4 self-employeed): "))
#health_issues = int(input("Enter health issues (0 for none, 1 for some issues): "))
#parent_marital_status = int(input("Enter parental marital status (0 for married, 1 for Single parrent, 2 for seperated): "))
#practice_sport = int(input("Enter practice sport frequency(0 for never, 1 for regular, 2 for sometimes): "))
#attendance = int(input("Enter attendance(o for excellent(<90%), 1 for good(<70%), 2 for low(40%)): "))
#homework_completion = int(input("Enter homework completion (0 for always complete, 1 for never complete, 2 for sometimes): "))
#academic_score = int(input("Enter academic score(0 for excellent(<90%), 1 for good(60%-89%), 2 for low(40%-59%), 3 for very poor(>30%)): "))
#attentivity_in_class = int(input("Enter attentivity in class (0 for excellent, 1 for good, 2 for low): "))
#behavioral_patterns = int(input("Enter behavioral patterns (0 for aggressive, 1 for calm, 2 for focused, 3 for hyperactive, 4 for sleepy): "))
#self_esteem = int(input("Enter self-esteem (0 for confident, 1 for low, 2 for over-confident): "))
#socially_skills = int(input("Enter socially skills (0 for active, 1 for inactive, 2 for moderately active): "))
#teacher_interaction = int(input("Enter teacher interaction (0 for good, 1 for moderate, 2 for nill): "))
#cognitive_development = int(input("Enter cognitive development (0 for excellent, 1 for good, 2 for low): "))
#technology_influence = int(input("Enter technology influence (0 for addicted, 1 for moderate, 2 for no usage): "))
#online_class_attentivity = int(input("Enter social media influence (0 for inactive, 1 for moderately active, 2 for over active): "))
#extra_curricular_involvement = int(input("Enter involvement in extra-curricular activities (0 for never, 1 for regularly, 2 for sometimes): "))

# Apply the input to the model
#test_vector = model_cat.predict([[age, gender, class_size, family_size, father_education, mother_education, 
#                                  father_occupation, mother_occupation, health_issues, parent_marital_status, 
#                                  practice_sport, attendance, homework_completion, academic_score, 
#                                  attentivity_in_class, behavioral_patterns, self_esteem, socially_skills, 
#                                  teacher_interaction, cognitive_development, technology_influence, 
#                                  online_class_attentivity, extra_curricular_involvement]])

# Interpret the prediction and provide output to the user
#print("---------------------")
#print('\n')
#if test_vector[0] == 0:
#    print("Average student")
#elif test_vector[0] == 1:
#    print("Below average student")
#elif test_vector[0] == 2:
#    print("Excellent student")
#elif test_vector[0] == 3:
#    print("Good")
#print('\n')
#print("---------------------")

