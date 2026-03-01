from asmagic import ARDataSubscriber
import numpy as np
import cv2

sub = ARDataSubscriber("192.168.31.13")

try:
    for data in sub:

        # Color image: process as numpy array
        if data.has_color_image:
            color = data.color_array  # RGB numpy array
            
            # Convert RGB to BGR for OpenCV
            bgr_image = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            
            # Display the image
            cv2.imshow("Camera RGB", bgr_image)
            
            # Wait for 1ms and check if 'q' is pressed to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    pass
finally:
    sub.close()
    cv2.destroyAllWindows()