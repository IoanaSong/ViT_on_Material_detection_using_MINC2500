#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
from transformers import ViTForImageClassification, ViTImageProcessor

class MaterialDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.yolo_model = YOLO('yolov5s.pt')  
        self.vit_model = ViTForImageClassification.from_pretrained('your_fine_tuned_model')
        self.vit_processor = ViTImageProcessor.from_pretrained('your_fine_tuned_model')
        
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)

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
    detector = MaterialDetector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()