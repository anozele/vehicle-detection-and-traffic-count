# vehicle-detection-and-traffic-counter
first by using haar cascade classifier in opencv ,we detected the vehicels and we get the width the height of our images,thus we can now get the centroid of our vhicle.
now after getting the centroid we can now count theno of vehicles gong through our cordinate.
But the problem with the opencv is that it has so many false detection.
so,to make our model good we useg Mogsubtractor in opencv,it has good accuracy ,but it will count if pepole also moved through it.



Now we used deep learning to classify the  vehicles into car,bus and bikes by using CNN in keras ,but the problem is that it cannot detect multiple obeject i
in a single frame.to makemultiple detection in single frame we need to use tensorflow object detection api.



so,first of all we need to collect the images for making tfrecords by labeling the images.
after converting the xml to csv and then to tfrecords ,we train our model from a pretrain model.I used frcnn model to train my model.after training 
the model we have to convert it into frozeninference_graph.pb for testing our model>
after that we can pass the video  through opencv and in each frame it will detect the vehicles and get the boundig box around it>
now after getting the bounding box we can now get the centorid of our object,and now count the traffic with high accuracy.



