import model
import cv2
import scipy
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "./save3/model.ckpt")

img = cv2.imread('./steering_wheel_image.jpg',0)
rows,cols = img.shape

smoothed_angle = 0

file_dataset = open("./data/indian_dataset/data.txt")
all_lines = file_dataset.readlines()

mean_square_error = 0
i = 1
while(cv2.waitKey(10) != ord('q')):
    
    full_image = cv2.imread("./data/indian_dataset/" + str(i) + ".jpg")
    
    image = cv2.resize(full_image[-150:], (120, 40),interpolation = cv2.INTER_AREA)/255
    degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / scipy.pi
    
    print("Predicted steering angle: " + str(degrees) + " degrees" + "   "+"Actual angle: "+all_lines[i-1].split(" ")[1]+" degrees")
    mean_square_error+= degrees*degrees
    cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
    
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("steering wheel", dst)
    i += 1

cv2.destroyAllWindows()
