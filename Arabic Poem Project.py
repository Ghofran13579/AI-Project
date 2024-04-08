# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 12:43:28 2024
@author: massa
"""

# Import des bibliothèques
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Chargement des données
df = pd.read_csv(r"C:\Users\massa\Downloads\archive (4)\Arabic_poetry_dataset.csv", encoding='utf-8')

# Prétraitement des données

# Création des ensembles de données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(df['poem_text'], df['category'], test_size=0.2, random_state=42)

# Extraction de caractéristiques
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Choix des classifieurs
nb_classifier = MultinomialNB()
rf_classifier = RandomForestClassifier()

# Entraînement des classifieurs
nb_classifier.fit(X_train_tfidf, y_train)
rf_classifier.fit(X_train_tfidf, y_train)

# Prédictions
nb_pred = nb_classifier.predict(X_test_tfidf)
rf_pred = rf_classifier.predict(X_test_tfidf)

# Évaluation des performances
print("Multinomial Naive Bayes Accuracy:", accuracy_score(y_test, nb_pred))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

# Autres mesures de performance
print("Multinomial Naive Bayes Classification Report:\n", classification_report(y_test, nb_pred))
print("Random Forest Classification Report:\n", classification_report(y_test, rf_pred))

# Cross-validation
nb_scores = cross_val_score(nb_classifier, X_train_tfidf, y_train, cv=10)
rf_scores = cross_val_score(rf_classifier, X_train_tfidf, y_train, cv=10)

print("Multinomial Naive Bayes Cross-Validation Scores:", nb_scores)
print("Random Forest Cross-Validation Scores:", rf_scores)
print("Average Multinomial Naive Bayes Cross-Validation Score:", nb_scores.mean())
print("Average Random Forest Cross-Validation Score:", rf_scores.mean())

# Classification d'un nouveau poème
new_poem = "يا طائر الشعر القرير يا وحي إلهام الصدور أسعف لساني برهة بالشعر قد قلّ النصير فلعله يُطفي به ما قد تأجج من سعير مجد العروبة ما له لا يستثار ولا يثور أتهاوناً قد صابهم أم صابهم يا ذا فتور مجد العروبة قد أقل عن عرشه بيد البنين وتقسمت أوطانهم بيد الطغاة الجائرين والذل سيطر فيهمو واحسرتا دنيا ودين والشرق أصبح خاضعاً يا للمجير وللمعين أبناؤه قد أسلموا بقيادهم للظالمين من بعد ما كان الأُلى بالسيف سادوا الخافقين داء التواكل قد فشا فينا فواعجز المراد وملوكنا قد فُخّمت أسماؤهم لكن جماد يرجون من أعدائهم أن يصلحوا لهم الفساد إن دام هذا حالنا فالبس على الشرق الحداد من بعد ما كان الأُلى بالسيف سادوا الخافقين بالشام كم كانت لهم من وقفة فيها جهار الروم شتت شملهم لا يملكون سوى الفرار والعرب سادوا أرضهم وتملكوا تلك الديار بالسيف والعدل معاً والحق بينهما منار وابن الوليد وكم أتى الله أكبر بانتصار فتحوا العواصم كلها فتحاً وقرّ لهم قرار كسرى وقيصر أصبحا من بعد عزٍّ في انكسار فهمو الذين همو همو بالسيف سادوا الخافقين أسطول مجدهم وما أدراك ما تلك الهمم قطع البحار ميمماً للغرب يخترق الخضم اسبانيا لا تحزني العلم جاءك والعلَمَ لله درك طارق حزت السباق على الأمم تركوا السفين محطماً كيما تجولُ به قدم هم الذين همو همو بالسيف سادوا الخافقين لذريق لم يرض الهوان ولم يجد إلاّ الجِلاد لم تمضِ إلا ساعة حتى تطهّرت البلاد الحق يرفع صوته الله ينصر من أراد نصر الاله جنوده فتعمرّت تلك البلاد رفعوا الفنون بعلمهم آثارهم حتى المعاد فهمو الذين همو همو بالسيف سادوا الخافقين لم يكفهم ما أحرزوا بالفتح والنصر المبين ثاروا بكل حميّة والله فوقهم معين أسأل فرنسا ما دهى أركانها تلك السنين تولوز كانت طعمة من سابق للفاتحين مع فرفشونه قد غدت نيس وليس لها معين فهمو الذين همو همو بالسيف سادوا الخافقين واسأل أورالَّ وبوردو مع بيزانس ومع ليون ماذا أصاب ملوكها في ساعة مرت وحين وأسأل بواتييه وما يدعونه نهراً لرون فهمو الذين همو همو بالسيف سادوا الخافقين لم يبق من أمجادنا إلا المآثر للجدود هم أورثونا مجدهم شرفاً وقد أخذوا العهود واحسرتا قد ضاع ما قد خلّفوا فمتى يعود هيهات حتى أرضنا بالذلّ ترسف بالقيود أوطاننا من ذلّة أمست علينا كالحود من بعد ما كان الاُلى بالسيف سادوا الخافين"
# Vectorize the new poem using the same TF-IDF vectorizer
new_poem_tfidf = tfidf_vectorizer.transform([new_poem])

# Predict the category using both classifiers
nb_prediction = nb_classifier.predict(new_poem_tfidf)
rf_prediction = rf_classifier.predict(new_poem_tfidf)

# Print the predictions
print("Multinomial Naive Bayes Prediction:", nb_prediction)
print("Random Forest Prediction:", rf_prediction)
