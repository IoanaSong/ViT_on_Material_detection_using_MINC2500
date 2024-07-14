#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
from transformers import ViTForImageClassification, ViTImageProcessor
from transformers import AutoImageProcessor, AutoModelForImageClassification

# from ultralytics import RTDETR



# # Use a pipeline as a high-level helper OR Load model directly (BELOW)
# from transformers import pipeline

# pipe = pipeline("image-classification", model="ioanasong/vit-MINC-2500")


# Load model directly
processor = AutoImageProcessor.from_pretrained("ioanasong/vit-MINC-2500")
model = AutoModelForImageClassification.from_pretrained("ioanasong/vit-MINC-2500")



class YoloViTMaterialDetector:
    def __init__(self):
        self.bridge = CvBridge()
        # self.yolo_model = YOLO('yolov10n.pt')  # "Suitable for extremely resource-constrained environments" for object-detection
        self.yolo_model = YOLO('yolov10s.pt')  # "Balances speed and accuracy" for object-detection
        # self.rtdetr_model = RTDETR("rtdetr-l.pt") # for OD

            # results = model("image.jpg")
            # results[0].show()
        # self.vit_model = ViTForImageClassification.from_pretrained('vit-MINC-2500')
        self.vit_model = model
        # self.vit_processor = ViTImageProcessor.from_pretrained('vit-MINC-2500')
        self.vit_processor = processor
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback) # TODO check which subscriber is camera for '/camera/image_raw'

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # YOLO detection
        results = self.yolo_model(cv_image)
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                crop = cv_image[y1:y2, x1:x2]
                
                # Vision Transformer inference
                inputs = self.vit_processor(images=crop, return_tensors="pt")
                outputs = self.vit_model(**inputs)
                predicted_class = outputs.logits.argmax(-1).item()
                
                # Draw bounding box and label
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(cv_image, f"Class: {predicted_class}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Material Detection", cv_image)
        cv2.waitKey(1)

if __name__ == '__main__':
    rospy.init_node('material_detector', anonymous=True)
    detector = YoloViTMaterialDetector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()