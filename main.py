import cv2
import numpy as np
import matplotlib.pyplot as plt

from color_classification import HistogramColorClassifier

my_classifier = HistogramColorClassifier(channels=[0, 1, 2],
                                         hist_size=[128, 128, 128],
                                         hist_range=[0, 256, 0, 256, 0, 256],
                                         hist_type='HSV')

model_1 = cv2.imread('model1_blue.png') #Blue Poison Dart Frog
my_classifier.addModelHistogram(model_1, name="Blue Poison Dart Frog")

model_2 = cv2.imread('model2_red.png') #Strawberry Poison Dart Frog
my_classifier.addModelHistogram(model_2, name="Strawberry Poison Dart Frog")

model_3 = cv2.imread('model3_corr.png')
my_classifier.addModelHistogram(model_3, name="Corroboree Frog")

model_4 = cv2.imread('model4_golden.png') #Golden Poison Dart Frog
my_classifier.addModelHistogram(model_4, name="Golden Poison Dart Frog")


image = cv2.imread('test_golden.jpg') #test
comparison_array = my_classifier.returnHistogramComparisonArray(image,
                                                                method="intersection")
comparison_array1 = my_classifier.returnHistogramComparisonArray(image,
                                                                method="correlation")
comparison_array2 = my_classifier.returnHistogramComparisonArray(image,
                                                                method="chisqr")
comparison_array3 = my_classifier.returnHistogramComparisonArray(image,
                                                                method="bhattacharyya")

print("Comparison array for intersection method", comparison_array)
print("Comparison array for correlation method", comparison_array1)
print("Comparison array for chisqr method", comparison_array2)
print("Comparison array for bhattacharyya method", comparison_array3)

print("Your frog is: ", my_classifier.returnBestMatchName(image=image))

barWidth = 0.25
fig = plt.subplots(figsize=(10, 10))
interseption = my_classifier.returnHistogramComparisonProbability(image, method="intersection")
plt.bar(my_classifier.returnNameList(), interseption, color='r', width=barWidth)
plt.xlabel('Frog species', fontweight='bold', fontsize=15)
plt.ylabel('Probability', fontweight='bold', fontsize=15)
plt.title("Histogram Comparison Probability")
plt.show()