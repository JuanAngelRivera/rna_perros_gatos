import os
import cv2

cats = ['bengal', 'ragamuffin', 'sphynx', 'turkish_angora']
dogs = ['american_foxhound_dog', 'italian_pointing_dog', 'pyrenean_sheepdog_smooth_faced', 'xoloitzcuintle_dog']

def process_images(input_path, output_path):
    for archive in os.listdir(input_path):
        route = os.path.join(input_path, archive)
        img = cv2.imread(route)

        img = cv2.resize(img, (128, 128))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
        
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blur, 100, 200)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

        output = img.copy()

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            if area >= 500:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]

                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)

        os.makedirs(output_path, exist_ok = True)

        cv2.imwrite(os.path.join(output_path, archive), edges)
        cv2.imwrite(os.path.join(output_path, 'labeled_' + archive), output)


for i in range(4):
    process_images('dataset/cat/' + cats[i], 'processed/cat/' + cats[i])
    process_images('dataset/dog/' + dogs[i], 'processed/dog/' + dogs[i])    
